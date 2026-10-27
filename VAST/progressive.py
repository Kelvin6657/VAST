"""Five-stage training, input preparation, and three-seed aggregation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SEEDS = (42, 123, 2026)
STAGES = ('Supervised baseline', 'Baseline + core SSL', 'SSL + view masking',
          'SSL + view masking + LoRA', 'Full VAST')

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def summarize_values(values):
    if len(values) != 3 or not all(__import__('math').isfinite(v) for v in values):
        raise ValueError('Exactly three finite independent seed values required')
    return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values), n=3)

def environment():
    from importlib.metadata import version
    return dict(python=sys.version,packages={name:version(name) for name in
        ('torch','torchvision','monai','numpy','scipy','h5py','scikit-image','scikit-learn')})

def prepare(root):
    """Validate local inputs and record the training protocol."""
    import h5py
    import numpy as np
    root = root.resolve()
    files = {}
    sets = {}
    counts = {}
    for name in ('train_labeled', 'train_unlabeled', 'valid'):
        records = json.loads((root/'data'/f'{name}.json').read_text(encoding='utf-8-sig'))
        ids = []
        views = {}
        for record in records:
            for key in ('image','label'):
                if not record.get(key):
                    continue
                old = record[key].replace('\\','/')
                if '/data/' in old:
                    p = root/'data'/old.split('/data/',1)[1]
                elif old.startswith('data/'):
                    p = root/old
                else:
                    p = Path(old)
                    if not p.is_absolute():
                        p = root/p
                if not p.is_file():
                    raise FileNotFoundError(p)
                if not p.resolve().is_relative_to((root/'data').resolve()):
                    raise ValueError('Training/validation input outside FETUS data: '+str(p))
                if p.suffix.lower() not in ('.h5','.hdf5'):
                    raise ValueError('Unexpected FETUS input type: '+str(p))
                record[key] = str(p.resolve())
            image = Path(record['image'])
            ids.append(image.stem)
            with h5py.File(image, 'r') as f:
                shape = f['image'].shape
                v = int(np.asarray(f['view']).reshape(-1)[0])
                if v not in (1,2,3,4):
                    raise ValueError(f'Invalid view {image}: {v}')
                views[str(v)] = views.get(str(v),0)+1
            if name != 'train_unlabeled':
                with h5py.File(record['label'],'r') as f:
                    mask = f['mask'][:]
                    if mask.shape != shape[:2] or mask.min()<0 or mask.max()>14:
                        raise ValueError(f'Image-mask alignment/class error: {image}')
                    if f['label'].shape != (7,):
                        raise ValueError(f'Classification shape error: {image}')
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate image IDs in {name}')
        sets[name] = ids
        target = HERE/'prepared'/f'{name}.json'
        write_json(target, records)
        files[name] = digest(target)
        counts[name] = dict(n=len(ids),views=views)
    for a,b in (('train_labeled','valid'),('train_unlabeled','valid'),('train_labeled','train_unlabeled')):
        shared = sorted(set(sets[a]) & set(sets[b]))
        if shared:
            raise ValueError(f'Split overlap {a}/{b}: {shared}')
    source = {str(p.relative_to(HERE)):digest(p) for p in sorted((HERE/'runtime').rglob('*.py'))}
    source.update({n:digest(HERE/n) for n in ('progressive.py','core.py')})
    write_json(HERE/'prepared'/'protocol.json', dict(seeds=SEEDS, stages=STAGES,
        counts=counts, ordered_ids=sets, split_hashes=files, source_hashes=source,environment=environment(),
        pretrained_sha256=digest(root/'pretrained_weights'/'echocare_encoder.pth'),
        checkpoint_rule='(Dice_percent + NSD_percent)/2 + 100*MacroF1_at_0.5',
        validation_policy='Same oracle segmentation view mask and eligible classification slots in ALL stages',
        pos_weight_policy='Same original view-eligible labeled sampling in ALL stages',
        encoder_policy='Frozen original encoder in all stages; zero frozen LoRA residual stages 0-2',
        steps_per_epoch=counts['train_unlabeled']['n']//8, project_root=str(root)))
    for stage in range(5):
        for seed in SEEDS:
            write_json(HERE/'configs'/f'stage{stage}_seed{seed}.json',dict(stage=stage,seed=seed,name=STAGES[stage],
                components=dict(ssl=stage>=1,view_mask=stage>=2,lora=stage>=3,cutmix=stage==4,feature_perturbation=stage==4),
                input_size=[256,256],batch_size=8,epochs=120,optimizer='AdamW',
                backbone_lr=1e-4,head_lr=1e-3,weight_decay=.01,lr_schedule='polynomial power .9',
                classification_threshold=.5,segmentation_confidence=.9,classification_tau_pos=.8,classification_tau_neg=.8,
                precision='bf16 AMP',gradient_clip=1.,nonfinite_policy='fail and retain traceback',
                data_root='PROJECT_ROOT/data',same_labeled_schedule_steps=counts['train_unlabeled']['n']//8,
                augmentation='runtime/dataset/transform.py unchanged',
                validation_policy='Same oracle segmentation mask and classification eligible slots for all stages',
                full_loss_weights={'x_seg':1.,'x_cls':1.,'x_fp_cls':1.,'u_s1_seg':.25,'u_s2_seg':.25,
                    'u_w_fp_seg':.5,'u_s1_mix_seg':.25,'u_s2_mix_seg':.25,'u_pseudo_cls':1.,'u_w_mix_cls':.25,'u_w_mix_fp_cls':.5} if stage==4 else None,
                core_loss_weights={'x_seg':1.,'x_cls':1.,'u_s1_seg':.25 if stage>=1 else 0.,
                    'u_s2_seg':.25 if stage>=1 else 0.,'u_pseudo_cls':1. if stage>=1 else 0.} if stage<4 else None,
                protocol_sha256=digest(HERE/'prepared'/'protocol.json')))
    print(json.dumps(counts), flush=True)

def check_protocol(root):
    p = json.loads((HERE/'prepared'/'protocol.json').read_text(encoding='utf-8'))
    if environment()!=p['environment']:
        raise ValueError('Python/package environment changed after prepare')
    if str(root.resolve()) != p['project_root']:
        raise ValueError('Run prepare after uploading to HPC; paths are host specific')
    for name,h in p['source_hashes'].items():
        if digest(HERE/name)!=h:
            raise ValueError('Source changed since prepare: '+name)
    for name,h in p['split_hashes'].items():
        if digest(HERE/'prepared'/f'{name}.json')!=h:
            raise ValueError('Split changed: '+name)
    if digest(root/'pretrained_weights'/'echocare_encoder.pth') != p['pretrained_sha256']:
        raise ValueError('Pretrained weights changed')
    return p

def train(root, stage, seed, smoke=False, device_name='cuda:0'):
    import numpy as np
    import torch
    from monai.utils import set_determinism
    sys.path.insert(0,str(HERE/'runtime'))
    import step_1_unimatch_train as t
    from core import core_epoch, evaluate
    from model.Echocare import _LoRA_qkv
    protocol = check_protocol(root)
    out = HERE/('smoke' if smoke else 'runs')/f'stage{stage}_seed{seed}'
    if out.exists():
        raise FileExistsError(f'Preserve existing run; choose a fresh output package: {out}')
    out.mkdir(parents=True)
    try:
        if device_name.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no implicit CPU training')
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device_name.startswith('cuda'):
            torch.cuda.manual_seed_all(seed)
        # MONAI transforms own their RNG; seed it as well as Python/NumPy/Torch.
        set_determinism(seed=seed)
        import dataset.transform as transforms
        transforms.train_l_transforms.set_random_state(seed=seed)
        transforms.u_strong_transforms.set_random_state(seed=seed+1)
        torch.backends.cudnn.benchmark=True  # Keep the trainer's cuDNN benchmarking policy.
        argv = sys.argv
        sys.argv = ['trainer', '--model','echocare','--opt','echocare',
            '--train-epochs','120','--batch-size','8','--num-workers','1',
            '--ssl-ckpt',str(root/'pretrained_weights'/'echocare_encoder.pth'),
            '--train-labeled-json',str(HERE/'prepared'/'train_labeled.json'),
            '--train-unlabeled-json',str(HERE/'prepared'/'train_unlabeled.json'),
            '--valid-labeled-json',str(HERE/'prepared'/'valid.json'),
            '--save-path',str(out),'--amp','--amp-dtype','bf16','--grad-clip','1']
        args = t.parse_args(); sys.argv=argv
        args.stage=stage
        # Use 120 epochs for the progressive training protocol.
        args.no_hard_view_mask=False
        args.cutmix_mode='same_view'
        from torch.utils.data import DataLoader
        from dataset.fetus import FETUSSemiDataset
        device = torch.device(device_name)
        model = t.build_model(args,device)
        adapters = [m for m in model.modules() if isinstance(m,_LoRA_qkv)]
        if len(adapters)!=24:
            raise RuntimeError(f'Unexpected LoRA insertion count {len(adapters)}')
        if stage<3:
            for m in adapters:
                for layer in (m.linear_a_q,m.linear_b_q,m.linear_a_v,m.linear_b_v):
                    layer.requires_grad_(False)
                assert torch.count_nonzero(m.linear_b_q.weight)==0
                assert torch.count_nonzero(m.linear_b_v.weight)==0
        opt,lrs=t.build_optimizer(args,model)
        seg_allowed=t.build_seg_allowed_mat(device,t.DEFAULT_SEG_ALLOWED,4,15)
        cls_allowed=t.build_allowed_mat(device,t.DEFAULT_CLS_ALLOWED,4,7)
        ul=FETUSSemiDataset(args.train_unlabeled_json,'train_u',size=256)
        ll=FETUSSemiDataset(args.train_labeled_json,'train_l',size=256,n_sample=len(ul))
        vl=FETUSSemiDataset(args.valid_labeled_json,'valid')
        from core import worker_init
        make=lambda db:DataLoader(db,batch_size=8,shuffle=True,drop_last=True,
            num_workers=1,worker_init_fn=worker_init)
        l,u,um=make(ll),make(ul),make(ul)
        v=DataLoader(vl,batch_size=1,shuffle=False,num_workers=0)
        pos=t.compute_pos_weight_from_loader(l,cls_allowed,7,device).to(device)
        manifest=dict(stage=stage,stage_name=STAGES[stage],seed=seed,smoke=smoke,
            components=dict(ssl=stage>=1,view_mask=stage>=2,lora=stage>=3,cutmix=stage==4,feature_perturbation=stage==4),
            args=vars(args),protocol_sha256=digest(HERE/'prepared'/'protocol.json'),
            source_hashes=protocol['source_hashes'],split_hashes=protocol['split_hashes'],
            validation_ids=protocol['ordered_ids']['valid'],torch=str(torch.__version__),
            monai=__import__('monai').__version__,cuda=torch.version.cuda,
            device=str(device),pos_weight=pos.cpu().tolist(),
            trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            evaluation_policy=protocol['validation_policy'],best_checkpoint=None)
        write_json(out/'manifest.json',manifest)
        logger=t.setup_logger(str(out))
        writer=t.SummaryWriter(log_dir=str(out/'tb'))
        total=len(u)*120; step=0; best=-float('inf'); best_epoch=None
        start=time.time()
        if smoke:
            # One real optimizer step at full batch/size, then held-out validation.
            l=[next(iter(l))]
            if stage>=1:
                u=[next(iter(u))]; um=[next(iter(um))]
            v=[vl[i] for i in range(min(4,len(vl)))]
            v=[tuple(x.unsqueeze(0) for x in sample) for sample in v]
        for epoch in range(1 if smoke else 120):
            kwargs=dict(args=args,model=model,optimizer=opt,scaler=None,device=device,
                train_loader_l=l,train_loader_u=u,train_loader_u_mix=um,
                allowed_seg_mat=seg_allowed,allowed_cls_mat=cls_allowed,pos_weight=pos,
                loss_w=t.DEFAULT_LOSS_WEIGHTS,writer=writer,logger=logger,epoch=epoch,
                base_lrs=lrs,global_step=step,total_iters=total)
            step=(t.train_one_epoch if stage==4 else core_epoch)(**kwargs)
            val=evaluate(args,model,device,v,seg_allowed,t.DEFAULT_CLS_ALLOWED)
            write_json(out/'validation'/f'epoch_{epoch:03d}.json',val)
            score=val['score']
            if not np.isfinite(score):
                raise ValueError('Nonfinite validation score')
            is_best=score>best
            if is_best:
                best,best_epoch=score,epoch
            checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=None,
                epoch=epoch,previous_best=best,best_epoch=best_epoch,global_step=step,
                protocol_sha256=manifest['protocol_sha256'],stage=stage,seed=seed,
                validation=val,args=vars(args),smoke=smoke)
            torch.save(checkpoint,out/'latest.pth')
            if is_best:
                torch.save(checkpoint,out/'best.pth')
            logger.info(f'Epoch {epoch}: Dice={val["Dice"]:.4f} NSD={val["NSD"]:.4f} MacroF1={val["Macro-F1"]:.5f} Score={score:.5f}')
        writer.close()
        # Reload the saved best and independently reproduce its saved metrics.
        saved=torch.load(out/'best.pth',map_location=device,weights_only=False)
        model.load_state_dict(saved['model'],strict=True)
        reproduced=evaluate(args,model,device,v,seg_allowed,t.DEFAULT_CLS_ALLOWED)
        for k in ('Dice','NSD','Macro-F1','score'):
            if abs(reproduced[k]-saved['validation'][k])>1e-5:
                raise ValueError('Saved checkpoint metric mismatch: '+k)
        result=dict(status='smoke_passed' if smoke else 'completed',stage=stage,seed=seed,
            completed_epochs=1 if smoke else 120,best_epoch=best_epoch,
            checkpoint_sha256=digest(out/'best.pth'),protocol_sha256=manifest['protocol_sha256'],
            split_hashes=manifest['split_hashes'],elapsed_seconds=time.time()-start,
            **reproduced)
        write_json(out/'result.json',result)
        manifest['best_checkpoint']=dict(epoch=best_epoch,path=str(out/'best.pth'),sha256=result['checkpoint_sha256'])
        write_json(out/'manifest.json',manifest)
    except Exception:
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(out/'status.json',dict(status='failed',stage=stage,seed=seed,smoke=smoke))
        raise

def aggregate():
    summaries=[]; raw=[]; missing=[]; protocol_hash=digest(HERE/'prepared'/'protocol.json')
    for stage in range(5):
        rows=[]
        for seed in SEEDS:
            d=HERE/'runs'/f'stage{stage}_seed{seed}'; p=d/'result.json'
            if not p.exists():
                missing.append(str(p.relative_to(HERE)));continue
            r=json.loads(p.read_text(encoding='utf-8'))
            if (r['status']!='completed' or r['completed_epochs']!=120 or
                r['stage']!=stage or r['seed']!=seed or r['protocol_sha256']!=protocol_hash or
                r['checkpoint_sha256']!=digest(d/'best.pth')):
                raise ValueError('Run provenance/budget/checkpoint mismatch: '+str(p))
            m=json.loads((d/'manifest.json').read_text(encoding='utf-8'))
            protocol=json.loads((HERE/'prepared'/'protocol.json').read_text(encoding='utf-8'))
            if m['validation_ids']!=protocol['ordered_ids']['valid'] or r['split_hashes']!=protocol['split_hashes']:
                raise ValueError('Ordered validation population mismatch')
            saved=json.loads((d/'validation'/f'epoch_{r["best_epoch"]:03d}.json').read_text(encoding='utf-8'))
            for k in ('Dice','NSD','Macro-F1','score'):
                if abs(saved[k]-r[k])>1e-5:
                    raise ValueError('Result and best validation mismatch')
            rows.append(r);raw.append(r)
        summaries.append(dict(stage=stage,name=STAGES[stage],n=len(rows),
            metrics={k:summarize_values([r[k] for r in rows]) for k in ('Dice','NSD','Macro-F1')} if len(rows)==3 else None,
            per_view={v:{k:summarize_values([r['per_view'][v][k] for r in rows])
                for k in ('Dice','NSD','Macro-F1') if all(r['per_view'][v][k] is not None for r in rows)}
                for v in ('4CH','LVOT','RVOT','3VT')} if len(rows)==3 else None))
    write_json(HERE/'results'/'raw_seeds.json',raw)
    write_json(HERE/'results'/'summary.json',dict(stages=summaries,missing=missing,complete=not missing))
    lines=['| Stage | Configuration | n | Dice (%) | NSD (%) | Macro-F1 |','|---|---|---:|---:|---:|---:|']
    tex=['\\begin{tabular}{llrrr}','Stage & Configuration & Dice (\\%) & NSD (\\%) & Macro-F1 \\\\', '\\hline']
    for s in summaries:
        vals=[f'{s["metrics"][k]["mean"]:.4f} ± {s["metrics"][k]["sample_std"]:.4f}' if s['metrics'] else 'NA (incomplete)' for k in ('Dice','NSD','Macro-F1')]
        lines.append(f'| {s["stage"]} | {s["name"]} | {s["n"]} | '+ ' | '.join(vals)+' |')
        tex.append(f'{s["stage"]} & {s["name"]} & '+' & '.join(v.replace('±',r'$\pm$') for v in vals)+r' \\')
    (HERE/'results'/'main_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (HERE/'results'/'main_table.tex').write_text('\n'.join(tex+[r'\end{tabular}'])+'\n',encoding='utf-8')
    pv=['| Stage | View | n | Dice (%) | NSD (%) | Macro-F1 |','|---|---|---:|---:|---:|---:|']
    for s in summaries:
        for v in ('4CH','LVOT','RVOT','3VT'):
            vals=[f'{s["per_view"][v][k]["mean"]:.4f} ± {s["per_view"][v][k]["sample_std"]:.4f}'
                if s['per_view'] and k in s['per_view'][v] else 'NA' for k in ('Dice','NSD','Macro-F1')]
            pv.append(f'| {s["stage"]} | {v} | {s["n"]} | '+' | '.join(vals)+' |')
    (HERE/'results'/'per_view_table.md').write_text('\n'.join(pv)+'\n',encoding='utf-8')
    deltas=[]
    for a,b in zip(summaries,summaries[1:]):
        deltas.append(dict(transition=f'{a["stage"]} -> {b["stage"]}',
            changes={k:b['metrics'][k]['mean']-a['metrics'][k]['mean'] for k in ('Dice','NSD','Macro-F1')} if a['metrics'] and b['metrics'] else None))
    write_json(HERE/'results'/'increments.json',deltas)
    dl=['| Transition | Δ Dice (pp) | Δ NSD (pp) | Δ Macro-F1 |','|---|---:|---:|---:|']
    for d in deltas:
        dl.append('| '+d['transition']+' | '+' | '.join(f'{d["changes"][k]:.4f}' if d['changes'] else 'NA' for k in ('Dice','NSD','Macro-F1'))+' |')
    (HERE/'results'/'increments.md').write_text('\n'.join(dl)+'\n',encoding='utf-8')
    report=['# Controlled progressive ablation', 'Mean ± sample standard deviation (ddof=1); preassigned seeds 42, 123, 2026.',
        'Dice/NSD are percentages; Macro-F1 is a fraction. Oracle evaluation masks are identical in all stages.',
        'Historical single-run and historical LoRA comparisons are not imported.',
        'Incomplete stages do not have a three-seed mean or standard deviation.', '']
    for d in deltas:
        report.append(f'{d["transition"]}: '+(json.dumps(d['changes']) if d['changes'] else 'No component conclusion available: incomplete training.'))
    (HERE/'results'/'experiment_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('Complete:',not missing,'Missing:',len(missing))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','train','smoke','aggregate'))
    p.add_argument('--project-root',type=Path,default=HERE.parent)
    p.add_argument('--stage',type=int,choices=range(5))
    p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.command=='prepare':prepare(a.project_root)
    elif a.command=='aggregate':aggregate()
    else:
        if a.stage is None:p.error('--stage required')
        train(a.project_root,a.stage,a.seed,a.command=='smoke',a.device)

if __name__=='__main__':main()
