"""
Deep-learning model comparison: trains several CNN architectures on the SAME
Kaggle data and the SAME split, and compares them.

  Custom CNN (from scratch) · MobileNetV2 · EfficientNetB0 · ResNet50V2 · DenseNet121

  python train_dl_compare.py --task eye                      # all architectures
  python train_dl_compare.py --task both --archs mobilenetv2 efficientnetb0 custom_cnn
  python train_dl_compare.py --task eye --promote-best       # best one becomes the app's model

Use a GPU (Kaggle) – the big backbones are slow on a laptop CPU.

Outputs
  models/<task>_<arch>.keras             each trained model
  models/<task>_dl_metrics.json          comparison (shown on the About page)
  models/<task>_dl_comparison.png        accuracy / F1 / speed chart
  results/<task>_dl_comparison.md        table for the report
"""
import argparse
import json
import shutil
from pathlib import Path

import train_model as tm


def plot(rows, task, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r["model"] for r in rows]
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(x - 0.2, [r["accuracy"] * 100 for r in rows], 0.4, label="Accuracy %", color="#22D3EE")
    ax1.bar(x + 0.2, [r["f1"] * 100 for r in rows], 0.4, label="F1 %", color="#A78BFA")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x, names, rotation=20, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, [r["infer_ms"] for r in rows], "o--", color="#F59E0B", label="ms / image")
    ax2.set_ylabel("ms / image")
    ax1.set_title(f"{task}: deep learning architectures (held-out test set)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["eye", "yawn", "both"], default="eye")
    ap.add_argument("--archs", nargs="+", default=list(tm.ARCHS), choices=list(tm.ARCHS))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fine-tune-epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (quick tests only)")
    ap.add_argument("--promote-best", action="store_true",
                    help="copy the best architecture to <task>_model.keras so the app uses it")
    args = ap.parse_args()

    tm.tf.keras.utils.set_random_seed(tm.SEED)
    root = tm.get_dataset_root(args.data_dir)
    out, res = Path(args.out), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    for task in (["eye", "yawn"] if args.task == "both" else [args.task]):
        rows = []
        for arch in args.archs:
            s = tm.train_task(task, root, out, tm.DEFAULT_IMG_SIZE[task], args.batch, args.epochs,
                              args.fine_tune_epochs, 30, None if args.no_pretrained else "imagenet",
                              arch, tag=arch)
            pos = s["classes"][1]
            rep = s["classification_report"].get(pos, {})
            rows.append(dict(model=s["arch_name"], arch=arch, family="deep learning",
                             accuracy=s["classification_report"]["accuracy"],
                             precision=rep.get("precision"), recall=rep.get("recall"), f1=rep.get("f1-score"),
                             auc=s["test_metrics"].get("auc"), train_s=s["train_s"], infer_ms=s["infer_ms"],
                             size_kb=s["size_mb"] * 1000, params=s["params"],
                             confusion_matrix=s["confusion_matrix"]))
        best = max(rows, key=lambda r: (r["f1"] or 0, r["accuracy"]))
        summary = {"task": task, "best_dl": best["model"], "results": rows}
        (out / f"{task}_dl_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        plot(rows, task, out / f"{task}_dl_comparison.png")

        md = [f"# {task} – deep learning architectures", "",
              "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Params | Size (MB) | Train (s) | ms / image |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -(r["f1"] or 0)):
            star = " ⭐" if r is best else ""
            md.append(f"| {r['model']}{star} | {r['accuracy'] * 100:.1f}% | {(r['precision'] or 0) * 100:.1f}% | "
                      f"{(r['recall'] or 0) * 100:.1f}% | {(r['f1'] or 0):.3f} | {(r['auc'] or 0):.3f} | "
                      f"{r['params'] / 1e6:.2f} M | {r['size_kb'] / 1000:.1f} | {r['train_s']:.0f} | {r['infer_ms']:.1f} |")
        (res / f"{task}_dl_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n" + "\n".join(md))

        if args.promote_best:
            shutil.copy(out / f"{task}_{best['arch']}.keras", out / f"{task}_model.keras")
            shutil.copy(out / f"{task}_{best['arch']}_metrics.json", out / f"{task}_metrics.json")
            print(f"\n{best['model']} copied to {task}_model.keras – the app now uses it.")


if __name__ == "__main__":
    main()
