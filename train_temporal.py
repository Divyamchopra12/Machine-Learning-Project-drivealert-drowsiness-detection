"""
Temporal (sequence) models: learn drowsiness from 3-second sequences of face signals.

Models compared (same data, split by video so no person/clip leaks into the test set):
  LSTM · GRU · Bidirectional LSTM · 1D-CNN (Keras)  +  Random Forest on window statistics (baseline)

Data sources (pick one):
  --rldd <folder>     UTA Real-Life Drowsiness Dataset: <fold>/<subject>/0.mp4 = alert,
                      10.mp4 = drowsy (5 = low vigilance, skipped unless --rldd-include-5)
  --videos <folder>   your own clips + labels.csv with columns video,start_s,end_s,label
                      where label is "alert" or "drowsy"
  --synthetic 60      generated sequences – ONLY to test that the pipeline runs (not real accuracy)

  python train_temporal.py --rldd D:/UTA-RLDD
  python train_temporal.py --videos clips
  python train_temporal.py --synthetic 60

Outputs
  models/temporal_model.keras + temporal_meta.json   best model – used live by the app
  models/temporal_metrics.json                        comparison (About page)
  results/temporal_comparison.md                      table for the report
  data/sequences.npz                                  cached extracted sequences
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from temporal import FEATURES, FPS, WINDOW, vector_from_result, window_stats

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
SEED = 42


# ------------------------------------------------------------ extraction ----
def video_frames(path, width=480):
    """Yield (t, frame) sampled at FPS."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / FPS))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            if frame.shape[1] > width:
                frame = cv2.resize(frame, None, fx=width / frame.shape[1], fy=width / frame.shape[1])
            yield i / src_fps, frame
        i += 1
    cap.release()


def extract_video(path, label_fn, max_seconds=None):
    """Run the detector over a video -> (frames x features) array + per-frame labels."""
    from detector import DetectorConfig, DrowsinessDetector

    det = DrowsinessDetector(DetectorConfig(use_cnn=False, circadian_risk=False, time_on_task_min=0,
                                            eye_occlusion_check=False), None, None)
    det.temporal = None     # never use an old temporal model while extracting
    feats, labels = [], []
    for t, frame in video_frames(path):
        if max_seconds and t > max_seconds:
            break
        r = det.process(frame, timestamp=t)
        lab = label_fn(t)
        if r["calibrating"]:
            continue
        feats.append(vector_from_result(r))
        labels.append(-1 if lab is None else lab)
    det.close()
    return np.array(feats, np.float32), np.array(labels, np.int32)


def make_windows(feats, labels, stride=10):
    xs, ys = [], []
    for s in range(0, len(feats) - WINDOW + 1, stride):
        lab = labels[s:s + WINDOW]
        lab = lab[lab >= 0]
        if len(lab) < WINDOW * 0.8:
            continue
        xs.append(feats[s:s + WINDOW])
        ys.append(int(round(lab.mean())))
    return xs, ys


def from_rldd(root, include_5=False, max_seconds=None):
    X, Y, G = [], [], []
    vids = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    for i, v in enumerate(vids):
        stem = v.stem.split("_")[0]
        if stem not in ("0", "10") + (("5",) if include_5 else ()):
            continue
        label = 0 if stem == "0" else 1
        print(f"  [{i + 1}/{len(vids)}] {v.parent.name}/{v.name} -> {'drowsy' if label else 'alert'}")
        f, l = extract_video(v, lambda t, lab=label: lab, max_seconds)
        xs, ys = make_windows(f, l)
        X += xs; Y += ys; G += [str(v.parent)] * len(xs)   # group = subject
    return X, Y, G


def from_labelled_clips(folder, labels_csv):
    rows = {}
    with open(labels_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lab = r["label"].strip().lower()
            if lab in ("alert", "drowsy"):
                rows.setdefault(r["video"], []).append((float(r["start_s"]), float(r["end_s"]), int(lab == "drowsy")))
    X, Y, G = [], [], []
    for video, iv in rows.items():
        path = Path(folder) / video
        if not path.exists():
            print(f"  ! missing {path}")
            continue
        print(f"  {video}")
        f, l = extract_video(path, lambda t, iv=iv: next((lab for s, e, lab in iv if s <= t < e), None))
        xs, ys = make_windows(f, l)
        X += xs; Y += ys; G += [video] * len(xs)
    return X, Y, G


def synthetic(n_videos, rng):
    """Plausible fake signals, ONLY for checking that training and the app pipeline work."""
    X, Y, G = [], [], []
    for v in range(n_videos):
        drowsy = v % 2
        T = 600
        ear = rng.normal(1.45, 0.05, T)
        closed = np.zeros(T)
        blink_every = 30 if drowsy else 45
        dur = (4, 12) if drowsy else (1, 3)
        for s in range(rng.integers(0, 20), T, blink_every + rng.integers(-8, 9)):
            d = rng.integers(*dur)
            ear[s:s + d] = rng.normal(0.6, 0.08, len(ear[s:s + d]))
            closed[s:s + d] = 1
        mar = rng.normal(0.12, 0.03, T)
        if drowsy:
            for s in rng.integers(0, T - 30, 3):
                mar[s:s + 20] = rng.normal(0.75, 0.05, 20)
        pitch = np.cumsum(rng.normal(0.15 if drowsy else 0, 0.3, T)) % 25 / 30 if drowsy else rng.normal(0, 0.05, T)
        yaw = rng.normal(0, 0.08, T)
        roll = np.abs(rng.normal(0, 0.05, T))
        f = np.stack([ear, mar, pitch, yaw, roll, closed, (mar > 0.6).astype(float), np.ones(T)], 1).astype(np.float32)
        xs, ys = make_windows(f, np.full(T, drowsy))
        X += xs; Y += ys; G += [f"syn{v}"] * len(xs)
    return X, Y, G


# ---------------------------------------------------------------- models ----
def build(name, n_feat):
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input((WINDOW, n_feat))
    if name == "LSTM":
        x = layers.LSTM(64)(inp)
    elif name == "GRU":
        x = layers.GRU(64)(inp)
    elif name == "Bidirectional LSTM":
        x = layers.Bidirectional(layers.LSTM(48))(inp)
    elif name == "1D-CNN":
        x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out, name=name.replace(" ", "_"))
    m.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m


def metrics(y, prob):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    pred = (prob >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, prob) if len(set(y)) > 1 else None,
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rldd")
    src.add_argument("--videos")
    src.add_argument("--synthetic", type=int)
    src.add_argument("--cached", action="store_true", help="reuse data/sequences.npz")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--rldd-include-5", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None, help="only use the first N seconds of each video")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="models")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    cache = Path("data/sequences.npz")
    if args.cached:
        d = np.load(cache, allow_pickle=True)
        X, Y, G, source = list(d["X"]), list(d["Y"]), list(d["G"]), str(d["source"])
    else:
        print("Extracting sequences ...")
        if args.rldd:
            X, Y, G = from_rldd(args.rldd, args.rldd_include_5, args.max_seconds); source = "UTA-RLDD"
        elif args.videos:
            X, Y, G = from_labelled_clips(args.videos, args.labels or Path(args.videos) / "labels.csv"); source = "own clips"
        else:
            X, Y, G = synthetic(args.synthetic, rng); source = "SYNTHETIC (pipeline test only)"
        cache.parent.mkdir(exist_ok=True)
        np.savez_compressed(cache, X=np.array(X), Y=np.array(Y), G=np.array(G), source=source)
    X, Y, G = np.array(X, np.float32), np.array(Y, np.int32), np.array(G)
    if len(set(Y)) < 2 or len(set(G)) < 3:
        raise SystemExit("Need both alert and drowsy windows from at least 3 videos.")
    print(f"{len(X)} windows of {WINDOW} frames x {X.shape[2]} features · {len(set(G))} videos · "
          f"{(Y == 1).mean() * 100:.0f}% drowsy · source: {source}")

    # split by video/subject: test 20%, then validation 15% of the rest
    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, Y, G))
    tr2, va2 = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(X[tr_idx], Y[tr_idx], G[tr_idx]))
    va_idx, tr_idx = tr_idx[va2], tr_idx[tr2]
    mean = X[tr_idx].reshape(-1, X.shape[2]).mean(0)
    std = X[tr_idx].reshape(-1, X.shape[2]).std(0) + 1e-6
    norm = lambda a: (a - mean) / std  # noqa: E731
    xtr, xva, xte = norm(X[tr_idx]), norm(X[va_idx]), norm(X[te_idx])
    ytr, yva, yte = Y[tr_idx], Y[va_idx], Y[te_idx]
    cw = {0: len(ytr) / (2 * max((ytr == 0).sum(), 1)), 1: len(ytr) / (2 * max((ytr == 1).sum(), 1))}

    rows, models = [], {}
    for name in ["LSTM", "GRU", "Bidirectional LSTM", "1D-CNN"]:
        print(f"\n== {name}")
        m = build(name, X.shape[2])
        t0 = time.time()
        m.fit(xtr, ytr, validation_data=(xva, yva), epochs=args.epochs, batch_size=64, class_weight=cw, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                                       restore_best_weights=True)])
        train_s = time.time() - t0
        prob = m.predict(xte, verbose=0).ravel()
        t0 = time.perf_counter()
        for _ in range(20):
            m(xte[:1], training=False)
        r = dict(model=name, family="temporal deep learning", **metrics(yte, prob), train_s=train_s,
                 infer_ms=(time.perf_counter() - t0) / 20 * 1000, params=int(m.count_params()))
        rows.append(r)
        models[name] = m
        print(f"   acc {r['accuracy'] * 100:.1f}%  F1 {r['f1']:.3f}")

    rf = RandomForestClassifier(300, n_jobs=-1, class_weight="balanced", random_state=SEED)
    stats = lambda a: np.stack([window_stats(w) for w in a])  # noqa: E731
    t0 = time.time()
    rf.fit(stats(X[tr_idx]), ytr)
    rows.append(dict(model="Random Forest (window stats)", family="classical baseline",
                     **metrics(yte, rf.predict_proba(stats(X[te_idx]))[:, 1]), train_s=time.time() - t0,
                     infer_ms=None, params=None))

    deep = [r for r in rows if r["family"].startswith("temporal")]
    best = max(deep, key=lambda r: (r["f1"], r["accuracy"]))
    out, res = Path(args.out), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    models[best["model"]].save(out / "temporal_model.keras")
    meta = {"best_model": best["model"], "features": FEATURES, "window": WINDOW, "fps": FPS,
            "mean": mean.tolist(), "std": std.tolist(), "source": source,
            "windows": int(len(X)), "videos": int(len(set(G))), "test_windows": int(len(te_idx))}
    (out / "temporal_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    clean = [{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()} for r in rows]
    (out / "temporal_metrics.json").write_text(json.dumps({**meta, "results": clean}, indent=2), encoding="utf-8")

    md = ["# Temporal models – drowsy vs alert from 3-second sequences", "",
          f"Source: {source} · {len(X)} windows from {len(set(G))} videos · split by video", "",
          "| Model | Type | Accuracy | Precision | Recall | F1 | ROC-AUC | Params |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(clean, key=lambda r: -r["f1"]):
        auc = "—" if r["auc"] is None else f"{r['auc']:.3f}"
        params = "—" if r["params"] is None else f"{r['params']:,}"
        star = " ⭐" if r["model"] == best["model"] else ""
        md.append(f"| {r['model']}{star} | {r['family']} | {r['accuracy'] * 100:.1f}% | {r['precision'] * 100:.1f}% | "
                  f"{r['recall'] * 100:.1f}% | {r['f1']:.3f} | {auc} | {params} |")
    (res / "temporal_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nBest: {best['model']} -> models/temporal_model.keras (used live by the app)")
    if source.startswith("SYNTHETIC"):
        print("NOTE: synthetic data only proves the pipeline works – train on UTA-RLDD or your clips for real results.")


if __name__ == "__main__":
    main()
