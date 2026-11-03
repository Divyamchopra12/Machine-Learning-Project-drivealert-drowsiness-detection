"""
Night / infrared eye-state model.

A normal webcam sees nothing in a dark car, so night systems use infrared (IR) LEDs and an
IR camera. IR images are grey and look different from daylight colour photos, so the day CNN
(trained on the Kaggle Yawn-Eye dataset) can be less accurate on them.

This script trains a second eye CNN on the MRL Eye Dataset – 84,898 infrared eye images
(37 people, 3 IR sensors, with/without glasses, reflections, bad/good light) – and compares
it with the day model on the same IR test images.

Dataset: http://mrl.cs.vsb.cz/eyedataset  (also re-uploaded on Kaggle – search "MRL eye dataset").
Works with both layouts:
  * the original: mrlEyes_2018_01/s0001/s0001_00001_0_0_0_0_0_01.png
    (fields: subject, image, gender, glasses, EYE STATE (0 closed / 1 open), reflections, light, sensor)
  * Kaggle copies with folders named Open / Closed (Open-Eyes / Close-Eyes ...)

Run:
  python train_night.py --data-dir D:\\mrlEyes_2018_01               # local / Kaggle path
  python train_night.py --data-dir ... --per-class 8000 --epochs 6     # faster
  python train_night.py --data-dir ... --by-subject                    # people never shared train/test

Outputs:
  models/eye_night.keras            used live automatically for IR / dark frames
  models/eye_night_metrics.json
  models/eye_night_compare.json     day model vs night model on IR test images (About page)
  results/night_vs_day.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np

import train_model as tm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def label_of(path: Path):
    """0 = open, 1 = closed (same as the day model), or None if unknown."""
    for part in reversed(path.parts[:-1]):
        k = part.lower().replace("-", "_").replace(" ", "_")
        if k in ("open", "open_eyes", "opened", "open_eye", "awake"):
            return 0
        if k in ("closed", "close", "close_eyes", "closed_eyes", "close_eye", "closed_eye", "sleepy"):
            return 1
    f = path.stem.split("_")
    if len(f) >= 8 and f[0].lower().startswith("s") and f[4] in ("0", "1"):
        return 0 if f[4] == "1" else 1   # MRL: eye state 1 = open, 0 = closed
    return None


def subject_of(path: Path) -> str:
    f = path.stem.split("_")
    return f[0] if f and f[0].lower().startswith("s") and f[0][1:].isdigit() else path.parent.name


def stage(src: Path, dst: Path, per_class: int, seed=42):
    """Copy (or hard-link) a balanced sample into dst/open and dst/closed so train_model can read it."""
    items = {0: [], 1: []}
    for root, _, files in os.walk(src):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                p = Path(root) / f
                y = label_of(p)
                if y is not None:
                    items[y].append(p)
    if not items[0] or not items[1]:
        raise SystemExit(f"Could not find open and closed IR eye images under {src}")
    print(f"Found {len(items[0])} open and {len(items[1])} closed images")
    rnd = random.Random(seed)
    if dst.exists():
        shutil.rmtree(dst)
    for y, name in ((0, "open"), (1, "closed")):
        (dst / name).mkdir(parents=True)
        chosen = items[y] if per_class <= 0 else rnd.sample(items[y], min(per_class, len(items[y])))
        for i, p in enumerate(chosen):
            out = dst / name / f"{subject_of(p)}__{i:06d}{p.suffix.lower()}"
            try:
                os.link(p, out)          # instant, no extra disk space (same drive)
            except OSError:
                shutil.copy2(p, out)
    return dst


def evaluate(model, paths, labels, img_size):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    ds = tm.make_ds(paths, labels, img_size, 64, shuffle=False)
    prob = model.predict(ds, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    return dict(accuracy=float(accuracy_score(labels, pred)), precision=float(precision_score(labels, pred, zero_division=0)),
                recall=float(recall_score(labels, pred, zero_division=0)), f1=float(f1_score(labels, pred, zero_division=0)),
                auc=float(roc_auc_score(labels, prob)))


_random_split = tm.split


def subject_split(paths, labels, seed=42):
    """Split by person so the test set only has people the model never saw."""
    subj = np.array([Path(p).name.split("__")[0] for p in paths])
    people = sorted(set(subj))
    if len(people) < 5:
        print("Subject IDs not found in file names – using a random split instead")
        return _random_split(paths, labels)
    random.Random(seed).shuffle(people)
    n = len(people)
    te, va = set(people[: max(1, int(n * 0.15))]), set(people[max(1, int(n * 0.15)): max(2, int(n * 0.30))])
    m_te, m_va = np.isin(subj, list(te)), np.isin(subj, list(va))
    m_tr = ~(m_te | m_va)
    return (paths[m_tr], labels[m_tr]), (paths[m_va], labels[m_va]), (paths[m_te], labels[m_te])


def main():
    ap = argparse.ArgumentParser(description="Train the infrared (night) eye-state model on MRL Eye")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work", default="data/mrl_staged")
    ap.add_argument("--per-class", type=int, default=15000, help="images per class (0 = all)")
    ap.add_argument("--arch", default="mobilenetv2", choices=list(tm.ARCHS))
    ap.add_argument("--img-size", type=int, default=tm.DEFAULT_IMG_SIZE["eye"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fine-tune-epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--by-subject", action="store_true", help="split train/val/test by person")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (quick tests only)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    out, res = Path(args.out), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    staged = stage(Path(args.data_dir), Path(args.work), args.per_class)

    if args.by_subject:
        tm.split = subject_split            # train_task uses tm.split internally
    summary = tm.train_task("eye", staged, out, args.img_size, args.batch, args.epochs, args.fine_tune_epochs,
                            30, None if args.no_pretrained else "imagenet", args.arch, tag="night")

    # same IR test images for both models
    paths, labels = tm.collect_images(staged, "eye")
    _, _, (p_te, y_te) = tm.split(paths, labels)
    night = tm.keras.models.load_model(out / "eye_night.keras")
    rows = [dict(model=f"Night IR model ({tm.ARCHS[args.arch]}, MRL)", **evaluate(night, p_te, y_te, night.input_shape[1]))]
    day_path = out / "eye_model.keras"
    if day_path.exists():
        day = tm.keras.models.load_model(day_path)
        rows.insert(0, dict(model="Day model (Kaggle Yawn-Eye, colour)", **evaluate(day, p_te, y_te, day.input_shape[1])))
    else:
        print("No models/eye_model.keras – train the day model first to get the comparison row.")
    cmp = {"dataset": "MRL Eye Dataset (infrared)", "test_images": int(len(p_te)),
           "split": "by subject" if args.by_subject else "random stratified", "results": rows}
    (out / "eye_night_compare.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")

    md = ["# Day vs night (infrared) eye model", "",
          f"Test set: {len(p_te)} infrared images from the MRL Eye Dataset ({cmp['split']} split).", "",
          "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['accuracy'] * 100:.1f}% | {r['precision'] * 100:.1f}% | "
                  f"{r['recall'] * 100:.1f}% | {r['f1']:.3f} | {r['auc']:.3f} |")
    md += ["", "The live system switches to the night model automatically when the camera image is infrared "
           "(grey) or dark (night mode)."]
    (res / "night_vs_day.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved {out / 'eye_night.keras'} – restart the app to use it.  Test accuracy: "
          f"{summary['classification_report']['accuracy'] * 100:.1f}%")


if __name__ == "__main__":
    main()
