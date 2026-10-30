"""Create an annotated-only training manifest and point project configs to it."""
from pathlib import Path
import datetime
import json
import shutil

root = Path(__file__).resolve().parents[1]
source = root / 'data/annotations/instances_train.json'
destination = root / 'data/annotations/instances_train_annotated.json'
config = root / 'vendor/yolact/data/config.py'
if not source.is_file() or not config.is_file():
    raise SystemExit('Place this script in your main Yolact/scripts folder after COCO conversion.')
with source.open(encoding='utf-8') as file:
    coco = json.load(file)
annotated_ids = {annotation['image_id'] for annotation in coco['annotations']}
kept = [image for image in coco['images'] if image['id'] in annotated_ids]
removed = [image for image in coco['images'] if image['id'] not in annotated_ids]
if not kept:
    raise SystemExit('No annotated training images found. No files changed.')
filtered = dict(coco, images=kept)
old_name, new_name = 'instances_train.json', 'instances_train_annotated.json'
updates = {}
for path in (config, root / 'configs/default.yaml'):
    if not path.is_file():
        continue
    text = path.read_text()
    if old_name not in text and new_name not in text:
        raise SystemExit(f'Could not find training annotation path in {path}; no files changed.')
    updates[path] = text.replace(old_name, new_name)
backup = root / 'portability_backups' / ('annotated_only_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
backup.mkdir(parents=True)
if destination.exists():
    shutil.copy2(destination, backup / destination.name)
with destination.open('w', encoding='utf-8') as file:
    json.dump(filtered, file, indent=2)
for path, text in updates.items():
    target = backup / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    path.write_text(text)
print(f'Training images kept: {len(kept)}')
print(f'Training images excluded (no annotations): {len(removed)}')
print(f'Package instances retained: {len(filtered["annotations"])}')
print('Training manifest:', destination)
print('YOLACT and available project config updated.')
print('Original images, labels, and training JSON preserved. Validation/test unchanged.')
