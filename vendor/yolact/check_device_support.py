import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import argparse
import torch
from utils.device import select_device
parser=argparse.ArgumentParser()
parser.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"])
device=select_device(parser.parse_args().device)
print("Checking device:",device)
from data.config import cfg, yolact_base_config
from yolact import Yolact
from layers.modules import MultiBoxLoss
from layers.output_utils import postprocess
from utils.augmentations import FastBaseTransform
cfg.replace(yolact_base_config.copy({'num_classes':2,'max_size':64,'freeze_bn':True}))
cfg.mask_proto_debug=False
torch.set_num_threads(2)
net=Yolact().to(device);net.train()
x=torch.rand(1,3,64,64, device=device)
pred=net(x)
criterion=MultiBoxLoss(2, cfg.positive_iou_threshold,cfg.negative_iou_threshold,cfg.ohem_negpos_ratio)
targets=[torch.tensor([[.2,.2,.8,.8,0.]], device=device)]
masks=[torch.ones(1,64,64, device=device)]
losses=criterion(net,pred,targets,masks,[0]); loss=sum(losses.values()); assert torch.isfinite(loss)
loss.backward()
print('PASS forward, loss, backward',float(loss.detach()))
net.eval();net.detect.use_fast_nms=True
with torch.no_grad():
 pred=net(x); result=postprocess(pred,64,64,score_threshold=0)
 print('PASS prediction/postprocess', [tuple(a.shape) for a in result])
 print('PASS preprocessing',tuple(FastBaseTransform().to(device)(torch.rand(1,64,64,3,device=device)).shape))
import eval as evaluation
evaluation.parse_args(['--device',str(device)])
with torch.no_grad():
 pred=net(x)
 image=evaluation.prep_display(pred,x[0],64,64)
 print('PASS mask/box rendering',image.shape)
