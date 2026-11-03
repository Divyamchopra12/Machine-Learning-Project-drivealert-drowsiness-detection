"""
Tune the fusion rule and alarm timing on YOUR OWN labelled clips – so every number in the
detector is chosen from data, not by hand.

Uses the same clips and labels.csv as evaluate.py (see that file for how to label them).

What is tuned (validation = your clips):
  1. Fusion weight w and decision threshold tau in
         eyes closed  <=>  w * P_cnn(closed) + (1 - w) * [EAR < s * thr] >= tau
     and the EAR threshold scale s.                  -> frame-level F1 heat-map
  2. Seconds of continuous closure before the alarm  -> microsleep detection rate vs
     false alarms per hour (the trade-off curve examiners ask about: "why 1.5 s?")

Run:
  python tune_fusion.py --videos clips              # writes results/fusion_tuning.*
  python tune_fusion.py --videos clips --apply      # also saves the best values as the app's settings

Honest use: tune on some people and report final numbers with evaluate.py on OTHER people
(use --people to choose which videos to tune on), otherwise the result is optimistic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from detector import DetectorConfig, DrowsinessDetector
from evaluate import MICROSLEEP_S, VIDEO_EXTS, label_at, load_labels

W_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
T_GRID = np.round(np.arange(0.3, 0.81, 0.05), 2)
S_GRID = np.round(np.arange(0.85, 1.16, 0.05), 2)
D_GRID = np.round(np.arange(0.5, 3.01, 0.25), 2)


def extract(path: Path, intervals, eye_model, width=640):
    """One pass per video: per-frame EAR, personal threshold, CNN P(closed) and the label."""
    cfg = DetectorConfig(eye_mode="fusion", use_cnn=True, circadian_risk=False, time_on_task_min=0,
                         use_temporal=False, phone_detection=False)
    det = DrowsinessDetector(cfg, eye_model, None)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if frame.shape[1] > width:
            frame = cv2.resize(frame, None, fx=width / frame.shape[1], fy=width / frame.shape[1])
        r = det.process(frame, timestamp=t)
        if r["calibrating"] or not r["face_detected"] or r["ear"] is None:
            continue
        lab, _ = label_at(intervals, t)
        rows.append((t, r["ear"], r["ear_threshold"], r["eye_closed_prob"],
                     None if lab not in ("open", "closed") else lab == "closed"))
    cap.release()
    det.close()
    return np.array(rows, dtype=object), i / fps


def fused(rows, w, tau, s):
    ear, thr = rows[:, 1].astype(float), rows[:, 2].astype(float)
    p = np.array([np.nan if v is None else v for v in rows[:, 3]], float)
    ear_closed = (ear < s * thr).astype(float)
    if np.isnan(p).all():                       # no CNN -> EAR only
        return ear_closed >= 0.5
    p = np.where(np.isnan(p), ear_closed, p)
    return (w * p + (1 - w) * ear_closed) >= tau


def f1(y, pred):
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


def alarms_from(t, closed, min_s):
    """Alarm start times: eyes closed continuously for >= min_s (same logic as the detector)."""
    out, since, fired = [], None, False
    for ti, c in zip(t, closed):
        if c:
            since = ti if since is None else since
            if not fired and ti - since >= min_s:
                out.append(ti)
                fired = True
        else:
            since, fired = None, False
    return out


def main():
    ap = argparse.ArgumentParser(description="Tune fusion weights, thresholds and alarm timing")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--eye-model", default="models/eye_model.keras")
    ap.add_argument("--people", nargs="*", help="only tune on videos whose name contains one of these")
    ap.add_argument("--out", default="results")
    ap.add_argument("--apply", action="store_true", help="save the best values into data/settings.json")
    args = ap.parse_args()

    folder, out = Path(args.videos), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels) if args.labels else folder / "labels.csv")
    vids = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_EXTS and p.name in labels]
    if args.people:
        vids = [p for p in vids if any(k.lower() in p.name.lower() for k in args.people)]
    if not vids:
        raise SystemExit("No labelled videos found – see evaluate.py --make-template")

    data, durations = {}, {}
    for v in vids:
        print(f"[extract] {v.name}")
        data[v.name], durations[v.name] = extract(v, labels[v.name], args.eye_model)
    lab_rows = np.concatenate([d[np.array([r[4] is not None for r in d], bool)] for d in data.values() if len(d)])
    if not len(lab_rows):
        raise SystemExit("No labelled frames with a visible face")
    y = lab_rows[:, 4].astype(bool)
    has_cnn = any(r[3] is not None for r in lab_rows)
    print(f"{len(y)} labelled frames ({int(y.sum())} closed) · CNN probabilities: {'yes' if has_cnn else 'no'}")

    # 1. grid search: w x tau x s
    grid, best = [], None
    for s in S_GRID:
        for w in (W_GRID if has_cnn else [0.0]):
            for tau in (T_GRID if has_cnn else [0.5]):
                score, prec, rec = f1(y, fused(lab_rows, w, tau, s))
                grid.append((float(w), float(tau), float(s), score, prec, rec))
                if best is None or score > best[3]:
                    best = grid[-1]
    w_b, tau_b, s_b = best[:3]
    default = [g for g in grid if g[0] == 0.6 and g[1] == 0.5 and g[2] == 1.0] if has_cnn else \
              [g for g in grid if g[2] == 1.0]
    ear_only = max((g for g in grid if g[0] == 0.0 and g[2] == s_b), key=lambda g: g[3])
    cnn_only = max((g for g in grid if g[0] == 1.0 and g[2] == s_b), key=lambda g: g[3]) if has_cnn else None
    print(f"best: w={w_b} tau={tau_b} EAR scale={s_b} -> F1 {best[3]:.3f}")

    # 2. alarm duration trade-off with the best fusion rule
    episodes = sum(1 for v in data for s0, e0, lab, _ in labels[v] if lab == "closed" and e0 - s0 >= MICROSLEEP_S)
    hours = sum(durations.values()) / 3600
    curve = []
    for d in D_GRID:
        det_n = fa = 0
        for v, rows in data.items():
            if not len(rows):
                continue
            alarms = alarms_from(rows[:, 0].astype(float), fused(rows, w_b, tau_b, s_b), d)
            micro = [(s0, e0) for s0, e0, lab, _ in labels[v] if lab == "closed" and e0 - s0 >= MICROSLEEP_S]
            any_closed = [(s0, e0) for s0, e0, lab, _ in labels[v] if lab == "closed"]
            det_n += sum(1 for s0, e0 in micro if any(s0 <= a <= e0 + 0.5 for a in alarms))
            fa += sum(1 for a in alarms if not any(s0 <= a <= e0 + 0.5 for s0, e0 in any_closed))
        curve.append((float(d), det_n / max(episodes, 1), fa / max(hours, 1e-9)))
    # pick the longest delay that still catches >= 95% of microsleeps (fewest false alarms)
    ok = [c for c in curve if c[1] >= 0.95] or [max(curve, key=lambda c: c[1])]
    d_b = max(ok, key=lambda c: c[0])

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    if has_cnn:
        hm = np.array([[next(g[3] for g in grid if g[0] == w and g[1] == t and g[2] == s_b) for w in W_GRID]
                       for t in T_GRID])
        im = axes[0].imshow(hm, origin="lower", aspect="auto", cmap="viridis",
                            extent=[W_GRID[0] - .05, W_GRID[-1] + .05, T_GRID[0] - .025, T_GRID[-1] + .025])
        axes[0].scatter([w_b], [tau_b], c="red", marker="*", s=200, label=f"best F1 {best[3]:.3f}")
        axes[0].scatter([0.6], [0.5], c="white", marker="o", s=60, label="hand-picked (0.6, 0.5)")
        axes[0].set_xlabel("weight on CNN probability  w   (0 = EAR only, 1 = CNN only)")
        axes[0].set_ylabel("decision threshold  τ")
        axes[0].set_title(f"Frame-level F1 (EAR scale {s_b})")
        fig.colorbar(im, ax=axes[0])
        axes[0].legend(loc="upper right", fontsize=8)
    else:
        axes[0].plot(S_GRID, [next(g[3] for g in grid if g[2] == s) for s in S_GRID], "o-")
        axes[0].set_xlabel("EAR threshold scale")
        axes[0].set_ylabel("F1")
        axes[0].set_title("EAR-only tuning (no CNN model found)")
    ax2 = axes[1]
    ax2.plot([c[0] for c in curve], [c[1] * 100 for c in curve], "o-", color="tab:green", label="microsleeps caught (%)")
    ax2.set_xlabel("seconds of eye closure before the alarm")
    ax2.set_ylabel("microsleeps caught (%)", color="tab:green")
    ax3 = ax2.twinx()
    ax3.plot([c[0] for c in curve], [c[2] for c in curve], "s--", color="tab:red", label="false alarms / hour")
    ax3.set_ylabel("false alarms per hour", color="tab:red")
    ax3.set_ylim(bottom=0)
    ax2.axvline(d_b[0], color="grey", ls=":")
    ax2.set_title(f"Alarm delay trade-off (chosen {d_b[0]} s)")
    fig.tight_layout()
    fig.savefig(out / "fusion_tuning.png", dpi=130)
    Path("models").mkdir(exist_ok=True)
    fig.savefig(Path("models") / "fusion_tuning.png", dpi=130)

    res = {"frames": int(len(y)), "closed_frames": int(y.sum()), "videos": [v.name for v in vids],
           "best": {"fusion_weight": w_b, "fusion_threshold": tau_b, "ear_scale": s_b, "f1": best[3],
                    "precision": best[4], "recall": best[5]},
           "hand_picked": None if not default else {"f1": default[0][3], "precision": default[0][4], "recall": default[0][5]},
           "ear_only": {"f1": ear_only[3]}, "cnn_only": None if cnn_only is None else {"f1": cnn_only[3], "threshold": cnn_only[1]},
           "alarm_delay": {"chosen_s": d_b[0], "detection_rate": d_b[1], "false_alarms_per_hour": d_b[2],
                           "curve": [{"seconds": c[0], "detection_rate": c[1], "false_alarms_per_hour": c[2]} for c in curve]},
           "microsleep_episodes": episodes, "hours": round(hours, 3)}
    Path("models").mkdir(exist_ok=True)
    Path("models/fusion_tuning.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = ["# Fusion and alarm-timing tuning", "",
          f"{len(y)} labelled frames ({int(y.sum())} eyes closed) from {len(vids)} videos.", "",
          "| Rule | F1 |", "|---|---|",
          f"| EAR only (scale {s_b}) | {ear_only[3]:.3f} |"]
    if cnn_only:
        md.append(f"| CNN only (threshold {cnn_only[1]}) | {cnn_only[3]:.3f} |")
    if default:
        md.append(f"| Hand-picked fusion (w 0.6, τ 0.5) | {default[0][3]:.3f} |")
    md += [f"| **Tuned fusion (w {w_b}, τ {tau_b}, EAR scale {s_b})** | **{best[3]:.3f}** |", "",
           "## Alarm delay", "", "| Seconds closed | Microsleeps caught | False alarms / hour |", "|---|---|---|"]
    md += [f"| {c[0]} | {c[1] * 100:.0f}% | {c[2]:.1f} |" for c in curve]
    md += ["", f"Chosen: **{d_b[0]} s** – the longest delay that still catches ≥ 95% of microsleeps "
           f"({episodes} labelled episodes ≥ {MICROSLEEP_S} s), which keeps false alarms lowest.", "",
           "Note: these values were tuned on the clips above. Report final accuracy on different people "
           "with evaluate.py so the numbers are not optimistic."]
    (out / "fusion_tuning.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    if args.apply:
        sf = Path("data/settings.json")
        st = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
        det = st.setdefault("detector", {})
        det.update({"fusion_weight": w_b, "fusion_threshold": tau_b, "eye_closed_seconds": d_b[0]})
        if s_b != 1.0:
            det["ear_threshold"] = round(det.get("ear_threshold", DetectorConfig.ear_threshold) * s_b, 3)
        sf.parent.mkdir(exist_ok=True)
        sf.write_text(json.dumps(st, indent=2), encoding="utf-8")
        print(f"\nSaved to {sf} – restart the app. (EAR scale only affects the non-calibrated threshold.)")


if __name__ == "__main__":
    main()
