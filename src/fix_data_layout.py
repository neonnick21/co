import os
import shutil

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SPLITS = [('train', 'train.json'), ('valid', 'val.json'), ('test', 'test.json')]

for split, ann_name in SPLITS:
    src_img_dir = os.path.join(DATA_DIR, split)
    dst_img_dir = os.path.join(DATA_DIR, 'images', 'train' if split == 'train' else ('val' if split == 'valid' else 'test'))
    os.makedirs(dst_img_dir, exist_ok=True)
    # Move images
    for fname in os.listdir(src_img_dir):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            shutil.move(os.path.join(src_img_dir, fname), os.path.join(dst_img_dir, fname))
    # Move annotation
    src_ann = os.path.join(src_img_dir, '_annotations.coco.json')
    dst_ann_dir = os.path.join(DATA_DIR, 'annotations')
    os.makedirs(dst_ann_dir, exist_ok=True)
    dst_ann = os.path.join(dst_ann_dir, ann_name)
    if os.path.exists(src_ann):
        shutil.move(src_ann, dst_ann)
    print(f"Moved {split} images and annotation to {dst_img_dir} and {dst_ann}")
