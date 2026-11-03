"""
Grad-CAM explainability: where does the CNN look when it decides
"eyes closed" or "yawning"?

  python gradcam.py --model models/eye_model.keras --images path/to/Closed path/to/Open
  python gradcam.py --model models/yawn_model.keras --images path/to/yawn --n 8

Outputs (in results/gradcam/):
  gradcam_grid.png      original | heatmap overlay for each image, with the prediction
  <name>_gradcam.png    individual overlays

Red areas are the pixels that pushed the prediction most. For a good eye model
they should sit on the eyelids / eye opening, not on eyebrows or background.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def build_gradcam_model(model):
    """Split the trained model into: input -> last conv feature map, and feature map -> prediction.
    The augmentation layers are skipped (they are inactive at inference anyway)."""
    # MobileNetV2 backbone = the last nested model (the "augment" Sequential is a Model too)
    base = [l for l in model.layers if isinstance(l, keras.Model)][-1]
    inp = keras.Input(model.input_shape[1:])
    names = [l.name for l in model.layers]
    x = model.get_layer("preprocess" if "preprocess" in names else "mobilenet_preprocess")(inp)
    conv = base(x, training=False)
    y = conv
    after = False
    for layer in model.layers:
        if layer is base:
            after = True
            continue
        if after:
            y = layer(y, training=False) if isinstance(layer, keras.layers.Dropout) else layer(y)
    return keras.Model(inp, [conv, y])


def gradcam(grad_model, img):
    x = tf.convert_to_tensor(img[None].astype(np.float32))
    with tf.GradientTape() as tape:
        conv, pred = grad_model(x, training=False)
        score = pred[:, 0]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))           # importance of each channel
    cam = tf.reduce_sum(conv[0] * weights[0], axis=-1)
    cam = tf.nn.relu(cam).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, float(pred[0, 0])


def overlay(img, cam):
    heat = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (0.55 * img + 0.45 * heat).astype(np.uint8)


def collect(paths, n):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            imgs = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
            files += imgs[: max(1, n // max(len(paths), 1))]
        elif p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/eye_model.keras")
    ap.add_argument("--images", nargs="+", required=True, help="image files and/or folders")
    ap.add_argument("--n", type=int, default=12, help="max images")
    ap.add_argument("--out", default="results/gradcam")
    ap.add_argument("--labels", default=None, help="class names, e.g. open,closed (default from model name)")
    args = ap.parse_args()

    model = keras.models.load_model(args.model)
    size = int(model.input_shape[1])
    names = (args.labels.split(",") if args.labels else
             ["no_yawn", "yawn"] if "yawn" in Path(args.model).name else ["open", "closed"])
    grad_model = build_gradcam_model(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = collect(args.images, args.n)
    if not files:
        raise SystemExit("No images found.")
    tiles = []
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size))
        cam, p = gradcam(grad_model, img)
        ov = overlay(img, cam)
        label = f"{names[int(p >= 0.5)]} {max(p, 1 - p) * 100:.0f}%"
        cv2.imwrite(str(out / f"{f.stem}_gradcam.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        pair = np.hstack([img, ov])
        pair = cv2.resize(pair, (320, 160), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(pair, (0, 0), (320, 20), (0, 0, 0), -1)
        cv2.putText(pair, f"{f.name[:22]} -> {label}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(pair)
        print(f"{f.name}: {label}")

    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out / "gradcam_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\nSaved {out / 'gradcam_grid.png'}")


if __name__ == "__main__":
    main()
