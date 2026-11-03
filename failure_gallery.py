"""
Failure analysis: WHERE and WHY the CNN is wrong.

Takes the held-out test images, finds the ones the model got wrong (most confident mistakes
first), shows each with its Grad-CAM heat-map (where the CNN looked) and tags a likely cause:

  dark          mean brightness < 60
  overexposed   mean brightness > 200
  blurry        variance of the Laplacian < 60 (little sharp detail)
  low contrast  grey-level standard deviation < 25
  borderline    probability between 0.4 and 0.6 (half-closed eyes, mid-yawn)
  other         none of the above – look at the heat-map

Run:
  python failure_gallery.py --task eye [--n 16] [--data-dir <dataset>]
Outputs:
  results/<task>_failures.png + .md, models/<task>_failures.png + .json (About page)
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import train_model as tm
from gradcam import build_gradcam_model, gradcam, overlay


def causes(img_rgb: np.ndarray, prob: float) -> list[str]:
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    out = []
    if gray.mean() < 60:
        out.append("dark")
    if gray.mean() > 200:
        out.append("overexposed")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 60:
        out.append("blurry")
    if gray.std() < 25:
        out.append("low contrast")
    if 0.4 <= prob <= 0.6:
        out.append("borderline")
    return out or ["other"]


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM gallery of the model's mistakes")
    ap.add_argument("--task", choices=["eye", "yawn"], default="eye")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out, mdir = Path(args.out), Path("models")
    out.mkdir(parents=True, exist_ok=True)

    model = tm.keras.models.load_model(args.model or mdir / f"{args.task}_model.keras")
    size = model.input_shape[1]
    paths, labels = tm.collect_images(tm.get_dataset_root(args.data_dir), args.task)
    _, _, (p_te, y_te) = tm.split(paths, labels)
    prob = model.predict(tm.make_ds(p_te, y_te, size, 64, shuffle=False), verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    wrong = np.where(pred != y_te)[0]
    names = tm.CLASS_NAMES[args.task]
    print(f"{len(wrong)} of {len(y_te)} test images wrong ({len(wrong) / len(y_te) * 100:.1f}%)")

    # all mistakes: cause statistics; gallery: the most confident mistakes
    def load(p):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (size, size)).astype(np.float32)

    stats, items = Counter(), []
    order = wrong[np.argsort(-np.abs(prob[wrong] - 0.5))]
    for k, i in enumerate(order):
        img = load(p_te[i])
        c = causes(img, prob[i])
        stats.update(c)
        kind = "false alarm" if pred[i] == 1 else "missed"
        items.append(dict(file=Path(p_te[i]).name, true=names[y_te[i]], pred=names[pred[i]],
                          prob=round(float(prob[i]), 3), kind=kind, causes=c))
        if k >= 400:          # enough for the statistics
            break

    show = order[: args.n]
    if len(show):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gm = build_gradcam_model(model)
        cols = 4
        rows_n = int(np.ceil(len(show) / cols))
        fig, axes = plt.subplots(rows_n, cols * 2, figsize=(cols * 4.2, rows_n * 2.4))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for k, i in enumerate(show):
            img = load(p_te[i])
            cam, _ = gradcam(gm, img)
            r, c = divmod(k, cols)
            axes[r, 2 * c].imshow(img.astype(np.uint8))
            axes[r, 2 * c].set_title(f"true {names[y_te[i]]}\npred {names[pred[i]]} ({prob[i]:.2f})", fontsize=8,
                                     color="red")
            axes[r, 2 * c + 1].imshow(overlay(img.astype(np.uint8), cam))
            axes[r, 2 * c + 1].set_title(", ".join(items[k]["causes"]), fontsize=8)
        fig.suptitle(f"{args.task} model – most confident mistakes (left: image, right: Grad-CAM)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"{args.task}_failures.png", dpi=110)
        shutil.copy(out / f"{args.task}_failures.png", mdir / f"{args.task}_failures.png")

    fa = int((pred[wrong] == 1).sum())
    res = {"task": args.task, "test_images": int(len(y_te)), "wrong": int(len(wrong)),
           "error_rate": float(len(wrong) / len(y_te)), "false_alarms": fa, "missed": int(len(wrong)) - fa,
           "causes": dict(stats.most_common()), "examples": items[: args.n]}
    (mdir / f"{args.task}_failures.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = [f"# Failure analysis – {args.task} model", "",
          f"{len(wrong)} of {len(y_te)} test images misclassified ({res['error_rate'] * 100:.1f}%): "
          f"{fa} false alarms ({names[1]} predicted, actually {names[0]}), {len(wrong) - fa} missed.", "",
          "| Likely cause | Mistakes |", "|---|---|"] + [f"| {k} | {v} |" for k, v in stats.most_common()]
    md += ["", f"![failures]({args.task}_failures.png)", "",
           "Use this to explain limitations and next steps (e.g. more dark / blurry training images, "
           "an IR camera, or a longer time window so one bad frame does not cause an alarm)."]
    (out / f"{args.task}_failures.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
