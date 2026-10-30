"""Plot saved YOLACT history and compare test predictions with COCO ground truth.

Place beside fine_tune.py in Yolact/scripts. Does not train or update weights.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def make_plots(run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    history_path = run / 'history.csv'
    if history_path.exists():
        with history_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        points = [(float(r['epoch']), float(r['train_loss'])) for r in rows
                  if r.get('epoch') and r.get('train_loss')]
        points = [(x, y) for x, y in points if math.isfinite(y)]
        if points:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(*zip(*points), color='#137c8b', label='Training: epoch average')
            ax.set(xlabel='Epoch', ylabel='Total loss', title='Training loss')
            ax.grid(alpha=.2)
            final_path = run / 'final_validation_metrics.json'
            if final_path.exists():
                final = json.loads(final_path.read_text())
                if final.get('val_loss_annotated') is not None:
                    ax.scatter([final.get('epoch', points[-1][0])],
                               [final['val_loss_annotated']], color='#d36a35',
                               label='Final validation loss (annotated images)', zorder=3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / 'loss.png', dpi=180)
            plt.close(fig)
        lr_points = [(float(r['epoch']), float(r['lr'])) for r in rows if r.get('lr')]
        if lr_points:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.step(*zip(*lr_points), where='post')
            ax.set(xlabel='Epoch', ylabel='Learning rate', yscale='log', title='Learning-rate schedule')
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(output / 'learning_rate.png', dpi=180)
            plt.close(fig)
    else:
        print('No history.csv found; no epoch loss curve can be reconstructed.')
    final_path = run / 'final_validation_metrics.json'
    if final_path.exists():
        final = json.loads(final_path.read_text())
        fig, ax = plt.subplots(figsize=(9, 5))
        for kind, color in [('box', '#7462a5'), ('mask', '#137c8b')]:
            values = final[kind]
            points = sorted((float(k) / 100, float(v)) for k, v in values.items() if k != 'all')
            ax.plot(*zip(*points), marker='o', color=color,
                    label=f'{kind.capitalize()}: mAP50–95 = {values["all"]:.2f}%')
        ax.set(xlabel='IoU threshold', ylabel='AP (%)', ylim=(0, 100),
               title='Final validation AP across IoU thresholds')
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'final_validation_ap.png', dpi=180)
        plt.close(fig)
    print('Plots:', output)


def overlay(image, masks, boxes, labels):
    import cv2
    import numpy as np
    result = image.copy()
    colors = [(65, 190, 60), (210, 95, 45), (60, 145, 240), (200, 80, 180), (30, 200, 210)]
    for i, mask in enumerate(masks):
        chosen = np.asarray(mask) > .5
        color = np.asarray(colors[i % len(colors)])
        result[chosen] = (result[chosen] * .6 + color * .4).astype(np.uint8)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        color = colors[i % len(colors)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, label, (max(0, x1), max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return result


def visualize(args):
    # Import only for inference; plot-only mode needs no model installation.
    import fine_tune as training
    import cv2
    import numpy as np
    import torch
    from pycocotools.coco import COCO
    from layers.output_utils import postprocess
    from utils.augmentations import FastBaseTransform

    device = training.select_device(args.device)
    training.set_cfg(args.config)
    training.cfg.freeze_bn = True
    training.cfg.mask_proto_debug = False
    coco = COCO(str(args.annotations))
    ids = sorted(coco.imgs)
    random.Random(args.seed).shuffle(ids)
    ids = ids if args.count == 0 else ids[:args.count]
    model = training.Yolact()
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    model = model.to(device)
    model.eval()
    model.detect.use_fast_nms = True
    model.detect.use_cross_class_nms = False
    transform = FastBaseTransform().to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = []
    with torch.inference_mode():
        for number, image_id in enumerate(ids, 1):
            record = coco.imgs[image_id]
            image = cv2.imread(str(args.images / record['file_name']))
            if image is None:
                raise FileNotFoundError(args.images / record['file_name'])
            h, w = image.shape[:2]
            frame = torch.from_numpy(image).to(device).float()
            predictions = model(transform(frame.unsqueeze(0)))
            classes, scores, boxes, masks = postprocess(
                predictions, w, h, crop_masks=True, score_threshold=args.threshold)
            scores = scores[1] if isinstance(scores, list) else scores
            if classes.numel():
                indices = scores.argsort(descending=True)[:args.top_k]
                classes = classes[indices].cpu().numpy()
                scores = scores[indices].cpu().numpy()
                boxes = boxes[indices].cpu().numpy()
                masks = masks[indices].cpu().numpy()
                names = training.cfg.dataset.class_names
                labels = [f'{names[int(c)]}: {s:.2f}' for c, s in zip(classes, scores)]
                predicted = overlay(image, masks, boxes, labels)
                count = len(classes)
            else:
                predicted = image.copy()
                count = 0
            anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
            gt_masks, gt_boxes, gt_labels = [], [], []
            for ann in anns:
                gt_masks.append(coco.annToMask(ann))
                x, y, bw, bh = ann['bbox']
                gt_boxes.append([x, y, x + bw, y + bh])
                gt_labels.append(coco.cats[ann['category_id']]['name'])
            truth = overlay(image, gt_masks, gt_boxes, gt_labels)
            panels = []
            for panel, title in [(truth, f'Ground truth ({len(anns)} annotations)'),
                                 (predicted, f'Prediction ({count}), confidence > {args.threshold:g}')]:
                banner = np.full((44, w, 3), 245, dtype=np.uint8)
                cv2.putText(banner, title, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, .55, (25, 25, 25), 1)
                panels.append(np.concatenate([banner, panel], axis=0))
            destination = args.output / f'{image_id}_{Path(record["file_name"]).stem}.jpg'
            if not cv2.imwrite(str(destination), np.concatenate(panels, axis=1)):
                raise OSError(f'Could not save {destination}')
            metadata.append({'image_id': image_id, 'image': record['file_name'],
                             'annotations': len(anns), 'predictions_displayed': count,
                             'output': destination.name})
            print(f'{number}/{len(ids)} saved: {destination.name}', flush=True)
    (args.output / 'selection.json').write_text(json.dumps(
        {'seed': args.seed, 'confidence_threshold': args.threshold, 'images': metadata}, indent=2))
    print('Test comparisons:', args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/yolact_package_seg')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--config', default='package_seg_config')
    parser.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='mps')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/images/test')
    parser.add_argument('--annotations', type=Path, default=ROOT / 'data/annotations/instances_test.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--count', type=int, default=20, help='Random test images; 0 means all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=.3)
    parser.add_argument('--top-k', type=int, default=100)
    args = parser.parse_args()
    if args.count < 0 or args.top_k < 1 or not 0 <= args.threshold <= 1:
        parser.error('count >= 0, top-k >= 1, threshold between 0 and 1 required')
    args.run = args.run.expanduser().resolve()
    args.weights = (args.weights or args.run / 'final_model.pth').expanduser().resolve()
    args.output = (args.output or args.run / 'test_visualizations').expanduser().resolve()
    args.images = args.images.expanduser().resolve()
    args.annotations = args.annotations.expanduser().resolve()
    make_plots(args.run, args.run / 'plots')
    if not args.plots_only:
        visualize(args)


if __name__ == '__main__':
    main()
