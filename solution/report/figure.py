"""Dev-only: report figure for one validation query. Left: drawing with the query box and the reviewed annotation
(green = positive, red = negative or blank, untinted = unscored). Right: the same drawing with the prediction (blue).
Usage: figure.py QUERY_ID PRED_DIR OUT_PNG [--scale 0.25]"""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate import reviewed_domains  # noqa: E402

qid, pred_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
scale = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 0.25
row = next(e for e in json.load(open(ROOT / "dataset" / "val.json"))["examples"] if e["id"] == qid)
gray = np.array(Image.open(ROOT / "dataset" / row["image"]).convert("L"))
pos, known, blank, neg = reviewed_domains(row, ROOT / "dataset")[:4]
pred = np.array(Image.open(pred_dir / f"{qid}.png").convert("L")) > 127
sz = (round(gray.shape[1] * scale), round(gray.shape[0] * scale))
small = lambda m, interp: cv2.resize(m.astype(np.float32), sz, interpolation=interp)
base = np.repeat(small(gray / 255.0, cv2.INTER_AREA)[..., None], 3, -1)


def tint(img, mask, color, alpha=0.45):
    m = small(mask, cv2.INTER_AREA)[..., None]
    return img * (1 - alpha * m) + np.array(color) * alpha * m


left = tint(tint(base, pos, (0.1, 0.8, 0.2)), known & ~pos, (0.9, 0.15, 0.15))
right = tint(base, pred, (0.15, 0.35, 0.95), 0.55)
panels = []
x0, y0, x1, y1 = row["query_box"]
for img in (left, right):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([x0 * scale - 2, y0 * scale - 2, x1 * scale + 2, y1 * scale + 2], outline=(255, 140, 0), width=3)
    panels.append(im)
# query crop inset, 4x
crop = Image.fromarray(gray[max(y0 - 8, 0):y1 + 8, max(x0 - 8, 0):x1 + 8]).convert("RGB")
crop = crop.resize((crop.width * 4, crop.height * 4), Image.NEAREST)
W = panels[0].width * 2 + 24
canvas = Image.new("RGB", (W, panels[0].height + crop.height + 48), (255, 255, 255))
canvas.paste(panels[0], (0, 40)); canvas.paste(panels[1], (panels[0].width + 24, 40))
canvas.paste(crop, (0, panels[0].height + 44))
d = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=22)
d.text((4, 8), f"{qid}  annotation: green positive, red negative/blank, untinted unscored; orange = query box", fill=(0, 0, 0), font=font)
d.text((panels[0].width + 28, 8), "prediction (blue)", fill=(0, 0, 0), font=font)
d.text((crop.width + 12, panels[0].height + 48), f"query crop ({x1-x0}x{y1-y0} px, shown 4x)", fill=(0, 0, 0), font=font)
canvas.save(out)
print(out, canvas.size)
