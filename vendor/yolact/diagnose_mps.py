"""Run from vendor/yolact: python /path/to/diagnose_mps.py --device mps.

Writes flushed operation traces even when Apple's backend aborts the process.
Does not edit the model, configuration, dataset, or checkpoints.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import argparse
import platform
import sys
from pathlib import Path

# Import the existing patched YOLACT from the current working directory.
sys.path.insert(0, str(Path.cwd()))
import torch
import torchvision
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu','mps','cuda'], default='mps')
parser.add_argument('--size', type=int, default=64)
args = parser.parse_args()
print('Python:',sys.version,flush=True)
print('macOS:',platform.mac_ver(),'architecture:',platform.machine(),flush=True)
print('PyTorch:',torch.__version__,'torchvision:',torchvision.__version__,flush=True)
print('Device:',args.device,'size:',args.size,flush=True)
print('MPS available:',torch.backends.mps.is_available(),flush=True)
print('MPS fallback:',os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),flush=True)
device=torch.device(args.device)

def sync():
    if device.type=='mps': torch.mps.synchronize()
    elif device.type=='cuda': torch.cuda.synchronize()

def describe(x):
    if isinstance(x,torch.Tensor):
        return f'shape={tuple(x.shape)} stride={x.stride()} dtype={x.dtype} device={x.device} contiguous={x.is_contiguous()}'
    if isinstance(x,(tuple,list)):
        return '['+', '.join(describe(v) for v in x)+']'
    return str(type(x).__name__)

print('BEGIN standalone convolution',flush=True)
conv=torch.nn.Conv2d(3,64,7,stride=2,padding=3).to(device)
x=torch.rand(1,3,args.size,args.size,device=device)
y=conv(x);sync()
print('PASS standalone convolution:',describe(y),flush=True)
del conv,y
from data.config import cfg,yolact_base_config
from yolact import Yolact
from layers.modules import MultiBoxLoss
cfg.replace(yolact_base_config.copy({'num_classes':2,'max_size':args.size,'freeze_bn':True}))
cfg.mask_proto_debug=False
torch.set_num_threads(2)
net=Yolact().to(device)
net.train()

def before(name):
    def hook(module,inputs):
        sync()
        print('ENTER',name,type(module).__name__,describe(inputs),flush=True)
        if isinstance(module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
            print('  WEIGHTS',describe(module.weight),'groups=',module.groups,flush=True)
            if isinstance(module,torch.nn.Conv2d) and inputs[0].shape[1]!=module.weight.shape[1]*module.groups:
                raise RuntimeError('Python-level input/weight channel mismatch at '+name)
    return hook

def after(name):
    def hook(module,inputs,output):
        sync()
        print('EXIT',name,describe(output),flush=True)
    return hook

for name,module in net.named_modules():
    if not any(module.children()):
        module.register_forward_pre_hook(before(name))
        module.register_forward_hook(after(name))
print('BEGIN YOLACT forward',flush=True)
pred=net(x);sync()
print('PASS YOLACT forward',flush=True)
criterion=MultiBoxLoss(2,cfg.positive_iou_threshold,cfg.negative_iou_threshold,cfg.ohem_negpos_ratio)
targets=[torch.tensor([[.2,.2,.8,.8,0.]],device=device)]
masks=[torch.ones(1,args.size,args.size,device=device)]
print('BEGIN loss',flush=True)
losses=criterion(net,pred,targets,masks,[0]);loss=sum(losses.values());sync()
print('PASS loss:',float(loss.detach().cpu()),flush=True)
print('BEGIN backward',flush=True)
loss.backward();sync()
print('PASS backward',flush=True)
