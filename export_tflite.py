"""
Edge deployment: convert the Keras models to TensorFlow Lite and compare
size, speed and accuracy (for a Raspberry Pi / phone version of the project).

  python export_tflite.py                       # size + speed for eye and yawn models
  python export_tflite.py --data-dir <dataset>  # also accuracy on held-out test images

Creates in models/:
  <task>_model_fp32.tflite   plain conversion
  <task>_model_fp16.tflite   float16 weights (half size, same accuracy on GPU/most CPUs)
  <task>_model_int8.tflite   dynamic-range int8 quantization (about 4x smaller)
and writes results/tflite_benchmark.md + .json.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402


def convert(model, mode):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


class TFLiteModel:
    def __init__(self, path):
        self.it = tf.lite.Interpreter(model_path=str(path), num_threads=os.cpu_count())
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def __call__(self, x):
        res = []
        for img in x:
            self.it.set_tensor(self.inp["index"], img[None].astype(np.float32))
            self.it.invoke()
            res.append(self.it.get_tensor(self.out["index"])[0, 0])
        return np.array(res)


def bench(fn, x, runs):
    fn(x[:1])  # warm-up
    t0 = time.perf_counter()
    for i in range(runs):
        fn(x[i % len(x): i % len(x) + 1])
    return (time.perf_counter() - t0) / runs * 1000


def test_images(data_dir, task, size, limit):
    import train_model as tm

    paths, labels = tm.collect_images(Path(data_dir), task)
    _, _, (p_te, y_te) = tm.split(paths, labels)
    p_te, y_te = p_te[:limit], y_te[:limit]
    imgs = []
    for p in p_te:
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        imgs.append(tf.image.resize(img, [size, size]).numpy())
    return np.stack(imgs), y_te


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="models")
    ap.add_argument("--data-dir", default=None, help="dataset folder for an accuracy check")
    ap.add_argument("--limit", type=int, default=400, help="max test images for the accuracy check")
    ap.add_argument("--runs", type=int, default=50, help="single-image inferences for timing")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    mdir, out = Path(args.models), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for task in ("eye", "yawn"):
        kpath = mdir / f"{task}_model.keras"
        if not kpath.exists():
            continue
        print(f"\n== {task} model ==")
        model = keras.models.load_model(kpath)
        size = int(model.input_shape[1])
        if args.data_dir:
            x, y = test_images(args.data_dir, task, size, args.limit)
        else:
            x, y = np.random.uniform(0, 255, (8, size, size, 3)).astype(np.float32), None

        rows = []
        keras_fn = lambda b: model(b, training=False).numpy().ravel()  # noqa: E731
        k_lat = bench(keras_fn, x, args.runs)
        k_acc = float(((keras_fn(x) >= 0.5) == y).mean()) if y is not None else None
        rows.append(dict(format="Keras (.keras)", size_mb=kpath.stat().st_size / 1e6, latency_ms=k_lat, accuracy=k_acc))
        for mode in ("fp32", "fp16", "int8"):
            path = mdir / f"{task}_model_{mode}.tflite"
            path.write_bytes(convert(model, mode))
            m = TFLiteModel(path)
            lat = bench(m, x, args.runs)
            acc = float(((m(x) >= 0.5) == y).mean()) if y is not None else None
            rows.append(dict(format=f"TFLite {mode}", size_mb=path.stat().st_size / 1e6, latency_ms=lat, accuracy=acc))
            print(f"  {mode}: {rows[-1]['size_mb']:.2f} MB, {lat:.1f} ms" + (f", acc {acc * 100:.1f}%" if acc is not None else ""))
        report[task] = rows

    if not report:
        raise SystemExit("No trained models found in " + str(mdir))
    (out / "tflite_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# TensorFlow Lite export – size, speed and accuracy", ""]
    for task, rows in report.items():
        base = rows[0]["size_mb"]
        md += [f"## {task} model", "", "| Format | Size (MB) | Smaller by | Latency (ms/image) | Test accuracy |", "|---|---|---|---|---|"]
        for r in rows:
            acc = "—" if r["accuracy"] is None else f"{r['accuracy'] * 100:.1f}%"
            md.append(f"| {r['format']} | {r['size_mb']:.2f} | {base / r['size_mb']:.1f}× | "
                      f"{r['latency_ms']:.1f} | {acc} |")
        md.append("")
    (out / "tflite_benchmark.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
