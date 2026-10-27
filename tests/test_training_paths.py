"""Synthetic CPU branch checks, never scientific results or GPU smoke tests."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'VAST'))
sys.path.insert(0, str(ROOT/'VAST/runtime'))
import logging
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
import step_1_unimatch_train as t
from core import core_epoch,evaluate

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__();self.seg=nn.Conv2d(1,15,1);self.cls=nn.Linear(1,7)
        self.fp=nn.Dropout2d(.5);self.calls=[]
    def forward(self,x,need_fp=False):
        self.calls.append((len(x),need_fp))
        def outputs(z):return self.seg(z),self.cls(z.mean((2,3)))
        s,c=outputs(x)
        if need_fp:
            fs,fc=outputs(self.fp(x));return (s,fs),(c,fc)
        return s,c

class Writer:
    def add_scalar(self,*a,**k):pass

class TrainingTests(unittest.TestCase):
    def test_stage_paths_have_finite_update_and_correct_branch_use(self):
        torch.set_num_threads(2)
        for stage in range(5):
            with self.subTest(stage=stage):
                torch.manual_seed(42)
                model=SmallModel();opt=torch.optim.AdamW(model.parameters(),lr=.001)
                x=torch.rand(8,1,16,16);view=torch.zeros(8,1,dtype=torch.long)
                mask=torch.ones(8,16,16,dtype=torch.long);label=torch.zeros(8,7);label[:,0]=1
                box=torch.ones(8,16,16)
                u=(x.clone(),view.clone(),x.clone(),x.clone(),box,box)
                # Deliberately absent base view in donor batch: strict fallback is exercised.
                um=(x.clone(),torch.full_like(view,3),x.clone(),x.clone(),box,box)
                a=SimpleNamespace(stage=stage,amp=False,amp_dtype='bf16',grad_clip=1.,
                    pseudo_tau_pos=.8,pseudo_tau_neg=.8,conf_thresh=.9,seg_num_classes=15,
                    no_hard_view_mask=False,cutmix_mode='same_view',skip_nonfinite_loss=False,tb_iter_freq=20)
                kw=dict(args=a,model=model,optimizer=opt,scaler=None,device=torch.device('cpu'),
                    train_loader_l=[(x,view,mask,label)],train_loader_u=[u],train_loader_u_mix=[um],
                    allowed_seg_mat=t.build_seg_allowed_mat(torch.device('cpu'),t.DEFAULT_SEG_ALLOWED,4,15),
                    allowed_cls_mat=t.build_allowed_mat(torch.device('cpu'),t.DEFAULT_CLS_ALLOWED,4,7),
                    pos_weight=torch.ones(7),loss_w=t.DEFAULT_LOSS_WEIGHTS,writer=Writer(),
                    logger=logging.getLogger('cpu_contract'),epoch=0,base_lrs=[.001],global_step=0,total_iters=120)
                before=model.seg.weight.detach().clone()
                steps=(t.train_one_epoch if stage==4 else core_epoch)(**kw)
                self.assertEqual(steps,1)
                self.assertFalse(torch.equal(before,model.seg.weight))
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                if stage==0:self.assertEqual(model.calls,[(8,False)])
                if stage<4:self.assertTrue(all(not fp for _,fp in model.calls))
                if stage==4:self.assertTrue(any(fp for _,fp in model.calls))
                val=evaluate(a,model,torch.device('cpu'),[(x[:1],view[:1],mask[:1],label[:1])],kw['allowed_seg_mat'],t.DEFAULT_CLS_ALLOWED)
                self.assertTrue(np.isfinite(val['score']))
                self.assertEqual(val['n_images'],1)

if __name__=='__main__':unittest.main()
