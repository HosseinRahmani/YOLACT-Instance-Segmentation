"""Measure a few actual optimizer steps before choosing a training schedule.
Uses temporary model weights; does not overwrite the pretrained checkpoint.
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
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--warmup', type=int, default=1)
args = parser.parse_args()
if args.steps < 1 or args.warmup < 0:
    parser.error('steps must be positive and warmup non-negative')
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

import statistics
criterion = MultiBoxLoss(cfg.num_classes, cfg.positive_iou_threshold,
                         cfg.negative_iou_threshold, cfg.ohem_negpos_ratio)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,
                            momentum=0.9, weight_decay=0.0005)
times = []
for step in range(args.warmup + args.steps):
    synchronize(device)
    started = time.perf_counter()
    image, (target, masks, crowds) = dataset[step % len(dataset)]
    images = image.unsqueeze(0).to(device, dtype=torch.float32)
    targets = [torch.as_tensor(target, device=device, dtype=torch.float32)]
    instance_masks = [torch.as_tensor(masks, device=device, dtype=torch.float32)]
    optimizer.zero_grad(set_to_none=True)
    predictions = model(images)
    losses = criterion(model, predictions, targets, instance_masks, [crowds])
    loss = sum(losses.values())
    if not torch.isfinite(loss).all().item():
        raise RuntimeError('Non-finite loss; benchmark stopped.')
    loss.backward()
    optimizer.step()
    synchronize(device)
    seconds = time.perf_counter() - started
    label = 'warmup' if step < args.warmup else 'measured'
    print(f'Step {step+1} ({label}): {seconds:.2f}s | loss={loss.detach().item():.5f}', flush=True)
    if step >= args.warmup:
        times.append(seconds)
    del predictions, losses, loss, images, targets, instance_masks
median = statistics.median(times)
print(f'Median seconds per training step: {median:.2f}', flush=True)
print(f'Estimated hours per epoch at batch size 1: {median * len(dataset) / 3600:.2f}', flush=True)
print('Estimate excludes validation and checkpoint writing; a short sample can vary across images.', flush=True)
print('Benchmark finished. Temporary trained weights were discarded; no checkpoint was modified.', flush=True)
