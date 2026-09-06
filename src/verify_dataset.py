import cv2
import matplotlib.pyplot as plt
from pathlib import Path

CLASS_NAMES = ['head', 'helmet', 'person']       # matches data.yaml order exactly
COLORS = [(220, 50, 50), (50, 205, 50), (50, 130, 220)]   # red, green, blue

def draw_boxes(img_path, lbl_path):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if Path(lbl_path).exists():
        with open(lbl_path) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                color = COLORS[int(cls) % len(COLORS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, CLASS_NAMES[int(cls)], (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def verify_split(img_dir, lbl_dir, n=9):
    imgs = list(Path(img_dir).glob("*.jpg"))[:n]
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for ax, p in zip(axes.flat, imgs):
        ax.imshow(draw_boxes(p, Path(lbl_dir) / (p.stem + ".txt")))
        ax.axis('off')
    plt.tight_layout()

    Path("notebooks").mkdir(exist_ok=True)
    plt.savefig("notebooks/verify.png", dpi=120)
    print("Saved → notebooks/verify.png ✓")

verify_split("Hard-Hat-Detection-1/train/images", "Hard-Hat-Detection-1/train/labels")