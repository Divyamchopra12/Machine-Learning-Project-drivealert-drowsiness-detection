"""
Is model A *really* better than model B, or just lucky on this test set?

For every trained eye/yawn model (main CNN, the 5 CNN architectures, best classical ML model)
this script predicts the SAME held-out test images and reports:

  * 95% bootstrap confidence intervals for accuracy and F1 (1,000 resamples of the test set)
  * McNemar's exact test against the reference model (the main CNN):
        b = images only the reference got right, c = images only the other model got right
        p-value from the binomial test on b vs c.  p < 0.05  -> the difference is significant.

Run:
  python significance.py --task eye  [--data-dir <dataset>]
Outputs:
  results/<task>_significance.md
  models/<task>_significance.json   (shown on the About page)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_model as tm

SEED = 42


def predict_keras(path, paths, labels):
    model = tm.keras.models.load_model(path)
    ds = tm.make_ds(paths, labels, model.input_shape[1], 64, shuffle=False)
    return model.predict(ds, verbose=0).ravel()


def predict_classical(path, paths):
    import joblib
    from train_classical import load_features

    bundle = joblib.load(path)
    return bundle["model"].predict_proba(load_features(paths))[:, 1], bundle["name"]


def f1_acc(y, pred):
    tp, fp, fn = ((y == 1) & (pred == 1)).sum(), ((y == 0) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return float((y == pred).mean()), float(2 * prec * rec / max(prec + rec, 1e-9))


def bootstrap(y, pred, n=1000):
    rng = np.random.default_rng(SEED)
    accs, f1s = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a, f = f1_acc(y[idx], pred[idx])
        accs.append(a)
        f1s.append(f)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return q(accs), q(f1s)


def mcnemar(y, ref, other):
    from scipy.stats import binomtest

    b = int(((ref == y) & (other != y)).sum())   # only the reference is right
    c = int(((ref != y) & (other == y)).sum())   # only the other model is right
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return b, c, p


def main():
    ap = argparse.ArgumentParser(description="Confidence intervals + McNemar test for all trained models")
    ap.add_argument("--task", choices=["eye", "yawn"], default="eye")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    mdir, out = Path(args.models), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    root = tm.get_dataset_root(args.data_dir)
    paths, labels = tm.collect_images(root, args.task)
    _, _, (p_te, y_te) = tm.split(paths, labels)        # same seed -> same test images as training
    print(f"{len(p_te)} test images")

    probs = {}
    main_path = mdir / f"{args.task}_model.keras"
    if main_path.exists():
        probs["Main CNN (MobileNetV2)"] = predict_keras(main_path, p_te, y_te)
    for arch, name in tm.ARCHS.items():
        f = mdir / f"{args.task}_{arch}.keras"
        if f.exists():
            probs[name] = predict_keras(f, p_te, y_te)
    cl = mdir / f"{args.task}_classical.joblib"
    if cl.exists():
        pr, name = predict_classical(cl, p_te)
        probs[f"{name} (classical, HOG)"] = pr
    if len(probs) < 2:
        raise SystemExit("Need at least two trained models (run train_model.py and train_classical.py / train_dl_compare.py)")

    preds = {k: (v >= 0.5).astype(int) for k, v in probs.items()}
    ref = next(iter(preds))
    rows = []
    for name, pred in preds.items():
        acc, f1 = f1_acc(y_te, pred)
        acc_ci, f1_ci = bootstrap(y_te, pred)
        b, c, p = (None, None, None) if name == ref else mcnemar(y_te, preds[ref], pred)
        rows.append(dict(model=name, accuracy=acc, accuracy_ci=acc_ci, f1=f1, f1_ci=f1_ci,
                         only_ref_right=b, only_this_right=c, p_value=p,
                         significant=None if p is None else p < 0.05))
        print(f"{name:<40} acc {acc * 100:5.1f}% [{acc_ci[0] * 100:.1f}-{acc_ci[1] * 100:.1f}]  "
              f"F1 {f1:.3f}" + ("" if p is None else f"  McNemar p={p:.4f}"))
    res = {"task": args.task, "test_images": int(len(y_te)), "reference": ref, "results": rows}
    (mdir / f"{args.task}_significance.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    md = [f"# Statistical comparison – {args.task} model", "",
          f"Same {len(y_te)} held-out test images for every model. 95% CI = bootstrap (1,000 resamples). "
          f"McNemar's exact test compares each model with **{ref}**.", "",
          "| Model | Accuracy (95% CI) | F1 (95% CI) | Only reference right | Only this right | McNemar p | Significant? |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['accuracy'] * 100:.1f}% ({r['accuracy_ci'][0] * 100:.1f}–{r['accuracy_ci'][1] * 100:.1f}) | "
                  f"{r['f1']:.3f} ({r['f1_ci'][0]:.3f}–{r['f1_ci'][1]:.3f}) | "
                  + ("— | — | — | reference |" if r["p_value"] is None else
                     f"{r['only_ref_right']} | {r['only_this_right']} | {r['p_value']:.4f} | {'yes' if r['significant'] else 'no'} |"))
    md += ["", "How to read it: if the confidence intervals overlap and p ≥ 0.05, the two models are not "
           "reliably different on this data – choose the faster / smaller one."]
    (out / f"{args.task}_significance.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved {out / (args.task + '_significance.md')}")


if __name__ == "__main__":
    main()
