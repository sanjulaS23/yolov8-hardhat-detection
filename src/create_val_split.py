import shutil
import random
from pathlib import Path

# ---- SETTINGS ----
dataset_root = Path("Hard-Hat-Detection-1")
val_ratio = 0.15  # 15% of train images go to validation
seed = 42
# ------------------

train_img_dir = dataset_root / "train" / "images"
train_lbl_dir = dataset_root / "train" / "labels"
val_img_dir   = dataset_root / "valid" / "images"
val_lbl_dir   = dataset_root / "valid" / "labels"

val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

all_images = list(train_img_dir.glob("*.jpg"))
random.seed(seed)
random.shuffle(all_images)

n_val = int(len(all_images) * val_ratio)
val_images = all_images[:n_val]

print(f"Total train images: {len(all_images)}")
print(f"Moving {n_val} images to valid/ ...")

for img_path in val_images:
    lbl_path = train_lbl_dir / (img_path.stem + ".txt")

    shutil.move(str(img_path), str(val_img_dir / img_path.name))
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(val_lbl_dir / lbl_path.name))

print("Done! ✓")
print(f"Remaining train images: {len(list(train_img_dir.glob('*.jpg')))}")
print(f"Valid images: {len(list(val_img_dir.glob('*.jpg')))}")