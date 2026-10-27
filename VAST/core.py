"""Stage 0-3 loss paths and one fixed evaluation protocol."""
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import step_1_unimatch_train as t

def worker_init(worker_id):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)
    from dataset import transform
    transform.train_l_transforms.set_random_state(seed=seed)
    transform.u_strong_transforms.set_random_state(seed=(seed+1)%2**32)

def core_epoch(**kw):
    a=kw['args'];model=kw['model'];opt=kw['optimizer'];device=kw['device']
    stage=a.stage;masking=stage>=2
    ce=torch.nn.CrossEntropyLoss();dice=t.DiceLoss(n_classes=15)
    steps=len(kw['train_loader_l']);step=kw['global_step']
    # Stage 0 reads NO unlabeled batches, even though the fixed schedule repeats labeled data.
    u_iter=iter(kw['train_loader_u']) if stage>=1 else None
    for i,(x,v,y,c) in enumerate(kw['train_loader_l']):
        x=x.to(device);v=v.to(device).long().view(-1);y=y.to(device);c=c.to(device).float()
        allowed=kw['allowed_cls_mat'][v] if masking else torch.ones_like(c)
        if stage>=1:
            w,vu,s1,s2,_,_=next(u_iter)
            w=w.to(device);vu=vu.to(device).long().view(-1);s1=s1.to(device);s2=s2.to(device)
            cm=kw['allowed_cls_mat'] if masking else torch.ones_like(kw['allowed_cls_mat'])
            with torch.no_grad():
                pt,conf_cls,_,_=t.teacher_pseudo(model,w,vu,kw['allowed_seg_mat'],cm,masking,a.pseudo_tau_pos,a.pseudo_tau_neg)
        model.train()
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=a.amp and device.type=='cuda'):
            if stage==0:
                seg,cls=model(x) # need_fp=False: no perturbation forward or loss
            else:
                weak_seg,weak_cls=model(torch.cat([x,w]))
                seg=weak_seg[:len(x)];cls=weak_cls[:len(x)]
                pseudo_logits=weak_seg[len(x):].detach()
                if masking:pseudo_logits=t.apply_view_mask_logits(pseudo_logits,vu,kw['allowed_seg_mat'])
                conf,pseudo=pseudo_logits.float().softmax(1).max(1)
                strong_seg,strong_cls=model(torch.cat([s1,s2]))
                sg1,sg2=strong_seg.chunk(2);cl1,cl2=strong_cls.chunk(2)
            if masking:seg=t.apply_view_mask_logits(seg,v,kw['allowed_seg_mat'])
            loss=(ce(seg,y)+dice(seg.softmax(1),y.unsqueeze(1).float()))/2
            loss=loss+t.masked_bce_with_logits(cls,c,allowed,pos_weight=kw['pos_weight'])
            if stage>=1:
                # Core weak-to-strong losses retain the corresponding original weights.
                loss=loss+.25*dice(sg1.softmax(1),pseudo.unsqueeze(1).float(),ignore=(conf<a.conf_thresh).float())
                loss=loss+.25*dice(sg2.softmax(1),pseudo.unsqueeze(1).float(),ignore=(conf<a.conf_thresh).float())
                loss=loss+.5*(t.masked_bce_with_logits(cl1,pt,conf_cls,pos_weight=kw['pos_weight'])+
                    t.masked_bce_with_logits(cl2,pt,conf_cls,pos_weight=kw['pos_weight']))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss stage={stage} epoch={kw["epoch"]} step={i}')
        opt.zero_grad(set_to_none=True);loss.backward()
        if a.grad_clip>0:torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip)
        opt.step();step+=1
        t.step_poly_lr(opt,kw['base_lrs'],kw['epoch']*steps+i,kw['total_iters'],power=.9)
        if i%20==0:
            kw['logger'].info(f'Stage {stage} epoch {kw["epoch"]} iter {i}/{steps}: loss={loss.item():.6f}')
            kw['writer'].add_scalar('train/loss',loss.item(),step)
    return step

@torch.no_grad()
def evaluate(args,model,device,loader,seg_allowed,cls_allowed):
    model.eval()
    sums=np.zeros((4,2,14),np.float64);counts=np.zeros((4,14),np.int64)
    positives=np.zeros((4,14),np.int64)
    yt=[];yp=[];views=[]
    for image,view,mask,label in loader:
        image=image.to(device);v=view.to(device).long().view(-1)
        if len(image)!=1:raise ValueError('Evaluation requires batch size 1')
        h,w=image.shape[-2:]
        seg,cls=model(F.interpolate(image,(256,256),mode='bilinear',align_corners=False))
        logits=F.interpolate(seg,(h,w),mode='bilinear',align_corners=False)
        # The same oracle evaluation in every stage; TRAIN masking is the ablated component.
        pm=t.apply_view_mask_logits(logits,v,seg_allowed).argmax(1).cpu().numpy()[0]
        gt=mask.numpy()[0];vi=int(v.item())
        if pm.shape!=gt.shape:raise ValueError('Prediction/GT resolution mismatch')
        for c in range(1,15):
            pb=pm==c;gb=gt==c;union=pb.sum()+gb.sum()
            positives[vi,c-1]+=int(gb.any())
            if union:
                sums[vi,0,c-1]+=2*(pb&gb).sum()/(union+1e-8)
                sums[vi,1,c-1]+=t.nsd_binary(pb,gb,tol=2.)
                counts[vi,c-1]+=1
        yt.append(label.numpy()[0]);yp.append(cls.sigmoid().cpu().numpy()[0]);views.append(vi)
    if not yt:raise ValueError('Empty validation set')
    yt=np.stack(yt);yp=np.stack(yp);views=np.asarray(views)
    metrics=t.masked_metrics_with_threshold_search(yt,yp,views,cls_allowed)
    count=counts.sum(0);total=sums.sum(0);valid=count>0
    vals=100*total/np.maximum(count,1)
    mean_dice=float(vals[0,valid].mean());mean_nsd=float(vals[1,valid].mean())
    f1=float(metrics['macro_f1@0.5'])
    per_view={}
    for vi,name in enumerate(('4CH','LVOT','RVOT','3VT')):
        eligible=counts[vi]>0;pv=100*sums[vi]/np.maximum(counts[vi],1)
        ix=views==vi
        fm=t.masked_metrics_with_threshold_search(yt[ix],yp[ix],views[ix],cls_allowed) if ix.any() else None
        class_slots=np.array(cls_allowed[vi],dtype=int)
        per_view[name]=dict(n_images=int(ix.sum()),
            Dice=float(pv[0,eligible].mean()) if eligible.any() else None,
            NSD=float(pv[1,eligible].mean()) if eligible.any() else None,
            **{'Macro-F1':float(np.mean(np.asarray(fm['per_class_f1@0.5'])[class_slots])) if fm else None},
            metric_counts=counts[vi].tolist(),gt_positive_counts=positives[vi].tolist())
    return dict(Dice=mean_dice,NSD=mean_nsd,**{'Macro-F1':f1},score=(mean_dice+mean_nsd)/2+100*f1,
        per_class_Dice=[float(vals[0,i]) if valid[i] else None for i in range(14)],
        per_class_NSD=[float(vals[1,i]) if valid[i] else None for i in range(14)],
        metric_counts=count.tolist(),gt_positive_counts=positives.sum(0).tolist(),
        missing_gt_classes=[i+1 for i in range(14) if positives[:,i].sum()==0],
        per_class_f1_at_0_5=metrics['per_class_f1@0.5'].tolist(),per_view=per_view,
        nsd_tolerance_pixels=2.,n_images=len(views))
