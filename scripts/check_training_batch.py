"""Check checkpoint transfer and one real augmented training image.
Run from the main project: python scripts/check_training_batch.py --device mps
No checkpoint, configuration, or dataset files are changed.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import argparse
from pathlib import Path
import random
import sys
import time

project = Path(__file__).resolve().parents[1]
source = project / 'vendor/yolact'
if not (source / 'yolact.py').is_file():
    raise SystemExit('Place this file in your main Yolact/scripts folder.')
sys.path.insert(0, str(source))
os.chdir(source)

import numpy as np
import torch
from data.config import cfg, set_cfg, MEANS
from data.coco import COCODetection
from utils.augmentations import SSDAugmentation
from utils.device import select_device, synchronize
from yolact import Yolact
from layers.modules import MultiBoxLoss

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', default='auto', choices=['auto','cpu','cuda','mps'])
parser.add_argument('--config', default='package_seg_config')
parser.add_argument('--pretrained', type=Path, default=Path('weights/yolact_base_54_800000.pth'))
args = parser.parse_args()
set_cfg(args.config)
cfg.freeze_bn = True  # batch size 1: use the pretrained BN statistics
cfg.mask_proto_debug = False
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = select_device(args.device)
print('Device:', device, '| Image size:', cfg.max_size, '| CPU operator fallback: disabled', flush=True)

checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=True)
model = Yolact()
current = model.state_dict()
compatible = {name: value for name, value in checkpoint.items()
              if name in current and value.shape == current[name].shape}
new_layers = [name for name in current if name not in compatible]
allowed = ('conf_layer','semantic_seg_conv','class_existence_fc')
unexpected = [name for name in new_layers if not any(part in name for part in allowed)]
if unexpected:
    raise RuntimeError('Checkpoint architecture mismatch: ' + repr(unexpected))
if not compatible:
    raise RuntimeError('No matching pretrained tensors were found.')
model.load_state_dict(compatible, strict=False)
print('Pretrained tensors loaded:', len(compatible), flush=True)
print('New class-dependent tensors:', ', '.join(new_layers), flush=True)
del checkpoint, current, compatible
model = model.to(device)
model.train()

dataset = COCODetection(cfg.dataset.train_images, cfg.dataset.train_info,
                        transform=SSDAugmentation(MEANS))
print('Training images exposed by loader:', len(dataset), flush=True)
if len(dataset) != len(dataset.coco.imgs):
    print('NOTE: this upstream loader excludes images without annotations; '
          'background-image handling needs adjustment before the full run.', flush=True)
image, (target, masks, crowds) = dataset[0]
images = image.unsqueeze(0).to(device, dtype=torch.float32)
targets = [torch.as_tensor(target, device=device, dtype=torch.float32)]
instance_masks = [torch.as_tensor(masks, device=device, dtype=torch.float32)]
print('Augmented batch:', tuple(images.shape), '| instances:', len(target), flush=True)
criterion = MultiBoxLoss(cfg.num_classes, cfg.positive_iou_threshold,
                         cfg.negative_iou_threshold, cfg.ohem_negpos_ratio)
synchronize(device)
start = time.perf_counter()
predictions = model(images)
losses = criterion(model, predictions, targets, instance_masks, [crowds])
loss = sum(losses.values())
if not torch.isfinite(loss).all().item():
    raise RuntimeError('Non-finite loss: ' + repr(losses))
loss.backward()
synchronize(device)
elapsed = time.perf_counter() - start
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        if parameter.grad.device.type != device.type:
            raise RuntimeError('Gradient on wrong device: ' + name)
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError('Non-finite gradient: ' + name)
print('PASS pretrained loading + augmented data + forward + loss + backward', flush=True)
print('Loss components:', {name: round(value.detach().item(), 5) for name, value in losses.items()}, flush=True)
print('One-batch forward/backward seconds:', round(elapsed, 2), flush=True)
print('No weights were saved or updated. This checks execution, not model accuracy.', flush=True)
