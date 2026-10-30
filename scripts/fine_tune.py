"""Fine-tune Package-Seg YOLACT. Run from project root with --device mps.
Resumes from the last completed epoch; augmentation is sampled at load time.
Validation runs once after all requested training epochs finish.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor/yolact'
sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)
import cv2
import numpy as np
import torch
from data.config import cfg, set_cfg, MEANS
from data.coco import COCODetection, detection_collate
from layers.modules import MultiBoxLoss
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.device import select_device, synchronize
from yolact import Yolact


class ValidationDataset(COCODetection):
    """Retain unannotated validation images for false-positive accounting."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ids = list(self.coco.imgs)

    def pull_item(self, index):
        image_id = self.ids[index]
        if self.coco.getAnnIds(imgIds=image_id):
            return super().pull_item(index)
        record = self.coco.imgs[image_id]
        image = cv2.imread(str(Path(self.root) / record['file_name']))
        if image is None:
            raise FileNotFoundError(record['file_name'])
        height, width = image.shape[:2]
        image, _, _, _ = self.transform(
            image, np.empty((0, height, width), np.float32),
            np.empty((0, 4), np.float32),
            {'labels': np.empty((0,), np.float32), 'num_crowds': 0})
        return (torch.from_numpy(image).permute(2, 0, 1),
                np.empty((0, 5), np.float32),
                np.empty((0, height, width), np.float32), height, width, 0)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def to_device(batch, device):
    images, (targets, masks, crowds) = detection_collate(batch)
    return (torch.stack(images).to(device),
            [target.to(device) for target in targets],
            [mask.to(device) for mask in masks], crowds)


def validate(model, criterion, dataset, device):
    # Loss is defined on annotated validation images; AP uses every image.
    model.train()  # frozen BN; expose the training prediction tensors
    loss_sum = 0.0
    annotated = 0
    with torch.no_grad():
        for index, image_id in enumerate(dataset.ids):
            if not dataset.coco.getAnnIds(imgIds=image_id):
                continue
            images, targets, masks, crowds = to_device([dataset[index]], device)
            losses = criterion(model, model(images), targets, masks, crowds)
            loss_sum += sum(losses.values()).item()
            annotated += 1
        model.eval()
        evaluation = importlib.import_module('eval')
        evaluation.parse_args(['--device', str(device), '--no_bar', '--max_images=-1'])
        original = evaluation.prep_metrics
        def metrics_with_empty_detections(ap_data, dets, img, gt, gt_masks,
                                          h, w, num_crowd, image_id, detections=None):
            detection = dets[0]['detection']
            if detection is None or detection['class'].numel() == 0:
                # Upstream returns early here and loses false negatives.
                classes = gt[:, 4].astype(int).tolist()
                if num_crowd:
                    classes = classes[:-num_crowd]
                for category in set(classes):
                    for kind in ('box', 'mask'):
                        for threshold in range(len(evaluation.iou_thresholds)):
                            ap_data[kind][threshold][category].add_gt_positives(classes.count(category))
                return
            if len(gt) == 0:
                from layers.output_utils import postprocess
                classes, scores, _, _ = postprocess(dets, w, h,
                    crop_masks=evaluation.args.crop, score_threshold=0)
                classes = classes.detach().cpu().tolist()
                score_sets = scores if isinstance(scores, list) else [scores, scores]
                for kind, values in zip(('box', 'mask'), score_sets):
                    values = values.detach().cpu().tolist()
                    for category, confidence in zip(classes, values):
                        for threshold in range(len(evaluation.iou_thresholds)):
                            ap_data[kind][threshold][category].push(confidence, False)
                return
            return original(ap_data, dets, img, gt, gt_masks, h, w,
                            num_crowd, image_id, detections)
        evaluation.prep_metrics = metrics_with_empty_detections
        try:
            maps = evaluation.evaluate(model, dataset, train_mode=True)
        finally:
            evaluation.prep_metrics = original
    if maps is None:
        raise RuntimeError('Validation interrupted before metrics were available.')
    model.train()
    return loss_sum / max(annotated, 1), maps



class InvalidTrainingBatch(RuntimeError):
    def __init__(self, reason, details):
        super().__init__(reason)
        self.details = details


def batch_sources(dataset, indices):
    return [{'index': i, 'image_id': int(dataset.ids[i]),
             'file_name': dataset.coco.imgs[dataset.ids[i]]['file_name']} for i in indices]


def check_finite_state(model, optimizer):
    tensors = [(f'model.{name}', value) for name, value in model.state_dict().items()
               if value.is_floating_point()]
    for i, entry in enumerate(optimizer.state.values()):
        tensors.extend((f'optimizer.{i}.{key}', value) for key, value in entry.items()
                       if torch.is_tensor(value) and value.is_floating_point())
    checks = [(name, torch.isfinite(value).all()) for name, value in tensors]
    if checks and not torch.stack([value for _, value in checks]).all().item():
        bad = [name for name, value in checks if not value.item()]
        raise RuntimeError('Non-finite model/optimizer state: ' + ', '.join(bad[:10]) +
                           '. Stopping; last completed-epoch checkpoint is preserved.')


def override_resume_lr(optimizer, scheduler, learning_rate):
    # Keep epoch milestones; apply their future decay factors to the new LR.
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
        group['initial_lr'] = learning_rate
    scheduler.base_lrs = [learning_rate for _ in optimizer.param_groups]
    scheduler._last_lr = [learning_rate for _ in optimizer.param_groups]


def accumulated_step(model, criterion, optimizer, dataset, indices, device, micro_batch_size, max_grad_norm=10.0):
    """Average microbatch losses by sample count; update once per group.

    YOLACT normalizes instance losses by positive anchors within each
    microbatch. Accumulation therefore does not exactly reproduce its
    physical-large-batch anchor normalization. Record the microbatch size.
    """
    optimizer.zero_grad(set_to_none=True)
    totals = {}
    for offset in range(0, len(indices), micro_batch_size):
        selected = indices[offset:offset+micro_batch_size]
        batch = [dataset[index] for index in selected]
        images, targets, masks, crowds = to_device(batch, device)
        valid_input = torch.isfinite(images).all()
        valid_targets = all(target.shape[0] > 0 and
                            torch.isfinite(target).all().item() and
                            ((target[:, 2:4] - target[:, :2]) > 0).all().item()
                            for target in targets)
        if not valid_input.item() or not valid_targets:
            raise InvalidTrainingBatch('Invalid augmented input or zero-area/empty targets',
                                       {'images': batch_sources(dataset, selected)})
        losses = criterion(model, model(images), targets, masks, crowds)
        loss = sum(losses.values())
        if not torch.isfinite(loss).all().item():
            raise InvalidTrainingBatch('Non-finite training loss',
                {'images': batch_sources(dataset, selected),
                 'losses': {key: str(value.detach().item()) for key, value in losses.items()}})
        weight = len(selected) / len(indices)
        (loss * weight).backward()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value.detach().item() * weight
        del batch, images, targets, masks, crowds, losses, loss
    # Check before optimizer.step so invalid gradients cannot update weights or momentum.
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                       error_if_nonfinite=True, foreach=False)
    except RuntimeError as exc:
        if 'non-finite' not in str(exc).lower():
            raise
        raise InvalidTrainingBatch('Non-finite gradient norm',
                                   {'images': batch_sources(dataset, indices)}) from exc
    optimizer.step()
    check_finite_state(model, optimizer)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto', choices=['auto','cpu','cuda','mps'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--micro-batch-size', type=int, default=None, help='Images processed together; gradients accumulate to --batch-size')
    parser.add_argument('--log-every', type=int, default=100, help='Print training progress every N optimizer batches')
    parser.add_argument('--max-skipped-batches', type=int, default=5, help='Stop if invalid batches exceed this count in one epoch; 0 stops at the first')
    parser.add_argument('--max-grad-norm', type=float, default=10.0, help='Clip gradient norm before each optimizer step')
    parser.add_argument('--resume-lr', type=float, default=None, help='Override saved LR when resuming, preserving future epoch decay milestones')
    parser.add_argument('--reset-momentum', action='store_true', help='Clear saved SGD momentum when resuming')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='package_seg_config')
    parser.add_argument('--pretrained', type=Path, default=Path('weights/yolact_base_54_800000.pth'))
    parser.add_argument('--output', type=Path, default=ROOT/'runs/yolact_package_seg')
    parser.add_argument('--resume', action='store_true', help='Resume from last completed epoch in output folder')
    args = parser.parse_args()
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error('max-grad-norm must be finite and positive')
    if args.resume_lr is not None and (not args.resume or not math.isfinite(args.resume_lr) or args.resume_lr <= 0):
        parser.error('resume-lr requires --resume and a finite positive value')
    if args.reset_momentum and not args.resume:
        parser.error('reset-momentum requires --resume')
    if args.max_skipped_batches < 0:
        parser.error('max-skipped-batches must be nonnegative')
    if args.log_every < 1:
        parser.error('log-every must be positive')
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size, and lr must be positive')
    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size
    if not 1 <= args.micro_batch_size <= args.batch_size:
        parser.error('micro-batch-size must be between 1 and batch-size')
    device = select_device(args.device)
    set_cfg(args.config)
    cfg.freeze_bn = True
    cfg.mask_proto_debug = False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    last = output/'last_training.pt'
    if last.exists() and not args.resume:
        raise RuntimeError('Output already contains training. Use --resume or a different --output folder.')
    model = Yolact()
    if not args.resume:
        source = torch.load(args.pretrained, map_location='cpu', weights_only=True)
        current = model.state_dict()
        matching = {k:v for k,v in source.items() if k in current and v.shape == current[k].shape}
        skipped = [k for k in current if k not in matching]
        if any(not any(part in k for part in ('conf_layer','semantic_seg_conv','class_existence_fc')) for k in skipped):
            raise RuntimeError('Checkpoint architecture does not match the selected config.')
        model.load_state_dict(matching, strict=False)
        print('Loaded pretrained tensors:', len(matching), '| initialized new class tensors:', skipped, flush=True)
        del source, current, matching
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=sorted(set([max(1, math.ceil(args.epochs*.6)), max(1, math.ceil(args.epochs*.8))])), gamma=.1)
    criterion = MultiBoxLoss(cfg.num_classes, cfg.positive_iou_threshold,
                            cfg.negative_iou_threshold, cfg.ohem_negpos_ratio)
    train = COCODetection(cfg.dataset.train_images, cfg.dataset.train_info, transform=SSDAugmentation(MEANS))
    val = ValidationDataset(cfg.dataset.valid_images, cfg.dataset.valid_info, transform=BaseTransform(MEANS))
    if len(train) != len(train.coco.imgs) or any(not train.coco.getAnnIds(imgIds=i) for i in train.ids):
        raise RuntimeError('Select instances_train_annotated.json for the training split.')
    print(f'Device: {device} | image size: {cfg.max_size} | train: {len(train)} | validation: {len(val)}', flush=True)
    print(f'Effective batch size: {args.batch_size} | physical microbatch: {args.micro_batch_size}', flush=True)
    print('On-the-fly training augmentation enabled. CPU operator fallback disabled.', flush=True)
    history=[]; best=-1.0; start_epoch=0
    if args.resume:
        state=torch.load(last,map_location='cpu',weights_only=False)
        if (state['settings']['epochs'],state['settings']['batch_size'],state['settings']['seed'],state['settings']['lr'],state['settings']['config']) != (args.epochs,args.batch_size,args.seed,args.lr,args.config):
            raise RuntimeError('Resume with the same epochs, batch-size, seed, lr, and config.')
        previous_micro = state['settings'].get('micro_batch_size', state['settings']['batch_size'])
        if previous_micro != args.micro_batch_size:
            raise RuntimeError('Resume with the same micro-batch-size.')
        if args.resume_lr is not None or args.reset_momentum:
            backup = output / f'last_before_recovery_epoch{state["epoch"]}.pt'
            if not backup.exists():
                shutil.copy2(last, backup)
            print('Recovery source checkpoint preserved:', backup, flush=True)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        if args.reset_momentum:
            optimizer.state.clear()
        finite = [torch.isfinite(t).all() for t in model.state_dict().values()
                  if t.is_floating_point()]
        if finite and not torch.stack(finite).all().item():
            raise RuntimeError('Checkpoint contains non-finite model tensors; recovery needs an earlier checkpoint.')
        print(f'Resuming after epoch {start_epoch if start_epoch else state["epoch"]}; replaying the incomplete epoch.', flush=True)
        for entry in optimizer.state.values():
            for key,value in entry.items():
                if torch.is_tensor(value): entry[key]=value.to(device)
        scheduler.load_state_dict(state['scheduler']); start_epoch=state['epoch'];best=state['best'];history=state['history']
        if args.resume_lr is not None:
            override_resume_lr(optimizer, scheduler, args.resume_lr)
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        if device.type=='mps' and state.get('mps_rng') is not None:torch.mps.set_rng_state(state['mps_rng'])
        del state
    check_finite_state(model, optimizer)
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]:g} | max gradient norm: {args.max_grad_norm:g}', flush=True)
    settings={**vars(args),'output':str(output),'pretrained':str(args.pretrained),'image_size':cfg.max_size,'train_images':len(train),'validation_images':len(val),'augmentation':'SSDAugmentation on load','batch_norm':'frozen','accumulation_loss':'sample-weighted mean of microbatch anchor-normalized losses','map_units':'percent','validation_loss':'annotated images only','validation_schedule':'once after all training epochs'}
    (output/'settings.json').write_text(json.dumps(settings,indent=2))
    try:
        for epoch in range(start_epoch,args.epochs):
            started=time.perf_counter();model.train();sums={};steps=0;seen=0;skipped=0
            order=list(range(len(train)));random.Random(args.seed+epoch).shuffle(order)
            lr=optimizer.param_groups[0]['lr']
            for position in range(0,len(order),args.batch_size):
                selected = order[position:position+args.batch_size]
                steps += 1
                try:
                    values = accumulated_step(model, criterion, optimizer, train, selected,
                                              device, args.micro_batch_size, args.max_grad_norm)
                except InvalidTrainingBatch as exc:
                    optimizer.zero_grad(set_to_none=True)
                    check_finite_state(model, optimizer)
                    skipped += 1
                    diagnostic = {'epoch': epoch+1, 'batch': steps, 'reason': str(exc),
                                  'skipped_in_epoch': skipped, **exc.details}
                    with (output/'invalid_batches.jsonl').open('a') as file:
                        file.write(json.dumps(diagnostic) + '\n')
                    print(f'WARNING epoch: {epoch+1} | batch: {steps} | skipped: {exc}'
                          f' | details: {output / "invalid_batches.jsonl"}', flush=True)
                    if skipped > args.max_skipped_batches:
                        raise RuntimeError('Too many invalid batches. Diagnostics saved; '
                                           'last_training.pt retains the last completed epoch.') from None
                    continue
                seen += len(selected)
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.0) + value * len(selected)
                total_steps = math.ceil(len(train) / args.batch_size)
                if steps == 1 or steps % args.log_every == 0:
                    components = ' | '.join(f'{key}={value:.4f}' for key, value in sorted(values.items()))
                    print(f'epoch: {epoch+1} | batch: {steps}/{total_steps}'
                          f' | loss={sum(values.values()):.4f} | {components}', flush=True)
            if not seen:
                raise RuntimeError('No valid training batches in this epoch.')
            scheduler.step();synchronize(device)
            record={'epoch':epoch+1,'lr':lr,'train_loss':sum(sums.values())/max(seen,1),
                    'trained_images':seen,'skipped_batches':skipped,
                    'seconds':time.perf_counter()-started}
            history.append(record)
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'history':history,'settings':settings,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None}
            atomic_save(state,last);atomic_save(model.state_dict(),output/'last_model.pth');del state
            with (output/'history.csv').open('w',newline='') as file:
                fields=list(dict.fromkeys(key for row in history for key in row))
                writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(history)
    except KeyboardInterrupt:
        print('Stopped. Resume replays the incomplete epoch from the last completed-epoch checkpoint.',flush=True)
        return
    atomic_save(model.state_dict(), output/'final_model.pth')
    print('Training finished. Computing final validation metrics...', flush=True)
    val_loss, maps = validate(model, criterion, val, device)
    metrics = {'epoch': args.epochs, 'val_loss_annotated': val_loss,
               'map_units': 'percent', 'box': maps['box'], 'mask': maps['mask']}
    (output/'final_validation_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'Validation loss (annotated images): {val_loss:.4f}', flush=True)
    print('Final model:', output/'final_model.pth', flush=True)
    print('Validation results:', output/'final_validation_metrics.json', flush=True)

if __name__=='__main__':main()
