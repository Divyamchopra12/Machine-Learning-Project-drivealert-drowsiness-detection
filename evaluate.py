"""
Ablation study on your own labelled videos: EAR-only vs CNN-only vs Fusion.

Modes compared: ear (EAR only), cnn (CNN only), fusion (CNN + EAR) and, if
train_classical.py has been run, classical (best classical ML model + EAR).

1. Record short webcam clips (normal, blinking, eyes closed 2-3 s, yawning,
   glasses, low light ...) and put them in a folder, e.g. clips/.
2. Create a label file:      python evaluate.py --videos clips --make-template
   This writes clips/labels.csv. Fill in the time ranges (seconds) when the
   eyes are CLOSED or OPEN. Optional "condition" column (normal, glasses,
   low_light, ...) gives a per-condition breakdown.

       video,start_s,end_s,label,condition
       me_normal.mp4,3.0,20.0,open,normal
       me_normal.mp4,20.0,23.5,closed,normal

3. Run:                      python evaluate.py --videos clips

Outputs (in results/):
  ablation.md / ablation.csv     frame-level accuracy, precision, recall, F1, latency
  conditions.md                  F1 per condition and mode
  events.md                      microsleep detection rate, delay, false alarms / hour

Frames in the first seconds (calibration) and frames without a label are skipped.
"""
import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from detector import DetectorConfig, DrowsinessDetector

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
MICROSLEEP_S = 1.5


def make_template(folder: Path):
    out = folder / "labels.csv"
    if out.exists():
        raise SystemExit(f"{out} already exists – edit it instead.")
    vids = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "start_s", "end_s", "label", "condition"])
        for v in vids:
            w.writerow([v, 3.0, 10.0, "open", "normal"])
            w.writerow([v, 10.0, 12.5, "closed", "normal"])
    print(f"Wrote {out} with example rows for {len(vids)} videos – edit the times and labels.")


def load_labels(path: Path):
    labels = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["video"]].append((float(row["start_s"]), float(row["end_s"]),
                                         row["label"].strip().lower(),
                                         (row.get("condition") or "all").strip() or "all"))
    return labels


def label_at(intervals, t):
    for s, e, lab, cond in intervals:
        if s <= t < e:
            return lab, cond
    return None, None


def run_video(path: Path, intervals, mode: str, eye_model, width=640):
    cfg = DetectorConfig(eye_mode=mode, use_cnn=mode != "ear", circadian_risk=False, time_on_task_min=0)
    det = DrowsinessDetector(cfg, eye_model if mode != "ear" else None, None)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, alarms, lat = [], [], []
    i, last_alarm = 0, False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if frame.shape[1] > width:
            frame = cv2.resize(frame, None, fx=width / frame.shape[1], fy=width / frame.shape[1])
        t0 = time.perf_counter()
        r = det.process(frame, timestamp=t)
        lat.append((time.perf_counter() - t0) * 1000)
        if r["alarm"] and r["alarm_reason"] == "eyes_closed" and not last_alarm:
            alarms.append(t)
        last_alarm = r["alarm"] and r["alarm_reason"] == "eyes_closed"
        lab, cond = label_at(intervals, t)
        if lab in ("open", "closed") and not r["calibrating"] and r["face_detected"]:
            frames.append((lab == "closed", r["eyes_closed"], cond))
    cap.release()
    det.close()
    return frames, alarms, lat, i / fps


def scores(pairs):
    y = np.array([p[0] for p in pairs], bool)
    p = np.array([p[1] for p in pairs], bool)
    tp, tn = int((y & p).sum()), int((~y & ~p).sum())
    fp, fn = int((~y & p).sum()), int((y & ~p).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(frames=len(y), accuracy=acc, precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def event_metrics(all_alarms, labels, durations):
    """Microsleep = labelled 'closed' interval >= 1.5 s. Detected if an eyes_closed alarm starts inside it."""
    episodes = detected = false_alarms = 0
    delays = []
    for video, alarms in all_alarms.items():
        closed = [(s, e) for s, e, lab, _ in labels[video] if lab == "closed" and e - s >= MICROSLEEP_S]
        episodes += len(closed)
        for s, e in closed:
            hits = [a for a in alarms if s <= a <= e + 0.5]
            if hits:
                detected += 1
                delays.append(hits[0] - s)
        any_closed = [(s, e) for s, e, lab, _ in labels[video] if lab == "closed"]
        false_alarms += sum(1 for a in alarms if not any(s <= a <= e + 0.5 for s, e in any_closed))
    hours = sum(durations.values()) / 3600
    return dict(episodes=episodes, detected=detected,
                detection_rate=detected / max(episodes, 1),
                mean_delay_s=float(np.mean(delays)) if delays else None,
                false_alarms=false_alarms, false_alarms_per_hour=false_alarms / max(hours, 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="folder with clips and labels.csv")
    ap.add_argument("--labels", default=None, help="label CSV (default: <videos>/labels.csv)")
    ap.add_argument("--eye-model", default="models/eye_model.keras")
    ap.add_argument("--out", default="results")
    ap.add_argument("--make-template", action="store_true")
    args = ap.parse_args()

    folder = Path(args.videos)
    if args.make_template:
        return make_template(folder)
    labels = load_labels(Path(args.labels) if args.labels else folder / "labels.csv")
    modes = ["ear"] + (["cnn", "fusion"] if Path(args.eye_model).exists() else [])
    if (Path(args.eye_model).parent / "eye_classical.joblib").exists():
        modes.append("classical")   # best classical ML model (Random Forest, SVM, ...) + EAR
    if len(modes) == 1:
        print("No eye model found – only the EAR-only mode will be evaluated.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, cond_rows, event_rows = [], [], []
    for mode in modes:
        pairs, alarms, lat, durations = [], {}, [], {}
        for video in labels:
            path = folder / video
            if not path.exists():
                print(f"  ! missing video {path}, skipped")
                continue
            print(f"[{mode}] {video}")
            f, a, l, d = run_video(path, labels[video], mode, args.eye_model)
            pairs += f
            alarms[video] = a
            lat += l
            durations[video] = d
        if not pairs:
            raise SystemExit("No labelled frames with a detected face – check labels.csv times.")
        s = scores(pairs)
        s.update(mode=mode, latency_ms=float(np.mean(lat)), fps=1000 / max(np.mean(lat), 1e-6))
        rows.append(s)
        for cond in sorted({p[2] for p in pairs}):
            cs = scores([p for p in pairs if p[2] == cond])
            cond_rows.append(dict(mode=mode, condition=cond, frames=cs["frames"], f1=cs["f1"], accuracy=cs["accuracy"]))
        event_rows.append(dict(mode=mode, **event_metrics(alarms, labels, durations)))

    keys = ["mode", "frames", "accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "latency_ms", "fps"]
    with (out / "ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)

    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731
    md = ["# Ablation study – eye-state detection", "",
          "| Mode | Frames | Accuracy | Precision | Recall | F1 | Latency (ms) | FPS |",
          "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['mode']} | {r['frames']} | {pct(r['accuracy'])} | {pct(r['precision'])} | {pct(r['recall'])} | "
           f"{r['f1']:.3f} | {r['latency_ms']:.1f} | {r['fps']:.1f} |" for r in rows]
    (out / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    cmd = ["# F1 by condition", "", "| Condition | Mode | Frames | Accuracy | F1 |", "|---|---|---|---|---|"]
    cmd += [f"| {c['condition']} | {c['mode']} | {c['frames']} | {pct(c['accuracy'])} | {c['f1']:.3f} |"
            for c in sorted(cond_rows, key=lambda c: (c["condition"], c["mode"]))]
    (out / "conditions.md").write_text("\n".join(cmd) + "\n", encoding="utf-8")

    emd = [f"# Microsleep detection (closed ≥ {MICROSLEEP_S}s)", "",
           "| Mode | Episodes | Detected | Detection rate | Mean delay (s) | False alarms | False alarms / hour |",
           "|---|---|---|---|---|---|---|"]
    for e in event_rows:
        delay = "—" if e["mean_delay_s"] is None else f"{e['mean_delay_s']:.2f}"
        emd.append(f"| {e['mode']} | {e['episodes']} | {e['detected']} | {pct(e['detection_rate'])} | "
                   f"{delay} | {e['false_alarms']} | {e['false_alarms_per_hour']:.1f} |")
    (out / "events.md").write_text("\n".join(emd) + "\n", encoding="utf-8")

    print("\n" + "\n".join(md) + "\n\n" + "\n".join(emd))
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
