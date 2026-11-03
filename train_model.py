"""
Train the CNN models for the Driver Drowsiness Detection System.

Dataset: Kaggle "Yawn-Eye Dataset" (serenaraju/yawn-eye-dataset-new)
  Folders: Open / Closed (eye crops), yawn / no_yawn (face images)

Two binary models are trained with transfer learning (default MobileNetV2;
--arch efficientnetb0 | resnet50v2 | densenet121 | custom_cnn also available):
  - eye  : open (0) vs closed (1)   -> models/eye_model.keras
  - yawn : no_yawn (0) vs yawn (1)  -> models/yawn_model.keras

Run on Kaggle (GPU on) or locally:
  python train_model.py --task both
  python train_model.py --task eye --data-dir /path/to/dataset
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

KAGGLE_DATASET = "serenaraju/yawn-eye-dataset-new"
KAGGLE_INPUT_DIR = "/kaggle/input/yawn-eye-dataset-new"

# Folder name (normalised) -> label. Label 1 is always the "drowsy" sign.
CLASS_MAP = {
    "eye": {"open": 0, "closed": 1},
    "yawn": {"no_yawn": 0, "yawn": 1},
}
CLASS_NAMES = {"eye": ["open", "closed"], "yawn": ["no_yawn", "yawn"]}
DEFAULT_IMG_SIZE = {"eye": 96, "yawn": 128}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SEED = 42


# ----------------------------------------------------------------- data ----
def _cached_dataset():
    """A copy kagglehub already extracted (~/.cache/kagglehub/datasets/<owner>/<name>/versions/N)."""
    base = Path.home() / ".cache" / "kagglehub" / "datasets" / KAGGLE_DATASET / "versions"
    versions = sorted((p for p in base.glob("*") if p.is_dir() and any(p.iterdir())), key=lambda p: p.name)
    return versions[-1] if versions else None


def get_dataset_root(data_dir=None):
    if data_dir:
        return Path(data_dir)
    if Path(KAGGLE_INPUT_DIR).exists():  # dataset attached in a Kaggle notebook
        return Path(KAGGLE_INPUT_DIR)
    cached = _cached_dataset()
    if cached:                            # already downloaded once - no second download
        print(f"Using cached dataset: {cached}")
        return cached
    import kagglehub  # downloads once, then cached

    print(f"Downloading {KAGGLE_DATASET} with kagglehub ...")
    try:
        return Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except PermissionError:
        # Windows: antivirus / OneDrive briefly locks the archive, so kagglehub cannot delete it
        # AFTER extracting. The data is already there - use it.
        cached = _cached_dataset()
        if cached:
            print(f"(archive locked by Windows, using the extracted copy) {cached}")
            return cached
        raise


def _norm(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def collect_images(root, task):
    """Walk the dataset and collect (path, label) for the task's class folders.
    Works whatever the train/test nesting is; duplicates are removed."""
    names = CLASS_MAP[task]
    paths, labels, seen = [], [], set()
    for dirpath, _, filenames in os.walk(root):
        key = _norm(Path(dirpath).name)
        if key not in names:
            continue
        for f in filenames:
            if not f.lower().endswith(IMG_EXTS):
                continue
            full = os.path.join(dirpath, f)
            ident = (key, f, os.path.getsize(full))
            if ident in seen:
                continue
            seen.add(ident)
            paths.append(full)
            labels.append(names[key])
    if not paths:
        raise SystemExit(
            f"No images found for task '{task}' under {root}. "
            f"Expected folders named {list(names)}."
        )
    return np.array(paths), np.array(labels, dtype=np.int32)


def split(paths, labels):
    from sklearn.model_selection import train_test_split

    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    p_val, p_te, y_val, y_te = train_test_split(
        p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    return (p_tr, y_tr), (p_val, y_val), (p_te, y_te)


def make_ds(paths, labels, img_size, batch, shuffle):
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        return img, tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- model ----
ARCHS = {
    # name: (display name, keras constructor or None for the custom CNN)
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNetB0",
    "resnet50v2": "ResNet50V2",
    "densenet121": "DenseNet121",
    "custom_cnn": "Custom CNN (from scratch)",
}


def _custom_backbone(img_size):
    """A small 4-block CNN trained from scratch (baseline for transfer learning)."""
    m = keras.Sequential(name="custom_backbone")
    m.add(keras.Input((img_size, img_size, 3)))
    for f in (32, 64, 128, 256):
        m.add(layers.Conv2D(f, 3, padding="same", use_bias=False))
        m.add(layers.BatchNormalization())
        m.add(layers.ReLU())
        m.add(layers.Conv2D(f, 3, padding="same", use_bias=False))
        m.add(layers.BatchNormalization())
        m.add(layers.ReLU())
        m.add(layers.MaxPooling2D())
    return m


def _backbone_and_preprocess(arch, img_size, weights):
    shape = (img_size, img_size, 3)
    if arch == "custom_cnn":
        return _custom_backbone(img_size), keras.Sequential([layers.Rescaling(1 / 255.0)], name="preprocess")
    if arch == "mobilenetv2":
        base = keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights=weights)
        pre = [layers.Rescaling(1 / 127.5, offset=-1.0)]
    elif arch == "resnet50v2":
        base = keras.applications.ResNet50V2(input_shape=shape, include_top=False, weights=weights)
        pre = [layers.Rescaling(1 / 127.5, offset=-1.0)]
    elif arch == "efficientnetb0":   # has its own rescaling inside
        base = keras.applications.EfficientNetB0(input_shape=shape, include_top=False, weights=weights)
        pre = [layers.Rescaling(1.0)]
    elif arch == "densenet121":      # "torch" mode: /255 then ImageNet mean/std
        base = keras.applications.DenseNet121(input_shape=shape, include_top=False, weights=weights)
        pre = [layers.Rescaling(1 / 255.0),
               layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])]
    else:
        raise ValueError(f"unknown arch {arch}; choose from {list(ARCHS)}")
    return base, keras.Sequential(pre, name="preprocess")


def build_model(img_size, weights="imagenet", arch="mobilenetv2"):
    """Input: raw RGB 0-255 images. Augmentation + preprocessing live inside
    the model, so inference code only needs to resize the crop."""
    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2, value_range=(0, 255)),
        ],
        name="augment",
    )
    base, preprocess = _backbone_and_preprocess(arch, img_size, weights)
    base.trainable = arch == "custom_cnn"   # pretrained backbones start frozen

    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = augment(inputs)
    x = preprocess(x)
    # pretrained backbones keep BatchNorm in inference mode, even while fine-tuning
    x = base(x) if arch == "custom_cnn" else base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob")(x)
    return keras.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


# ------------------------------------------------------------- training ----
def plot_history(histories, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = {}
    for h in histories:
        for k, v in h.history.items():
            hist.setdefault(k, []).extend(v)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(hist[metric], label="train")
        ax.plot(hist[f"val_{metric}"], label="val")
        if len(histories) > 1:
            ax.axvline(len(histories[0].history["loss"]) - 0.5, ls="--", c="grey",
                       label="fine-tune starts")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_confusion(cm, names, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def train_task(task, root, out_dir, img_size, batch, epochs, ft_epochs, ft_layers,
               weights="imagenet", arch="mobilenetv2", tag=None):
    """tag=None saves the main model (<task>_model.keras, used by the app);
    a tag such as "resnet50v2" saves <task>_resnet50v2.keras for the comparison."""
    import time
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"\n========== Training '{task}' model · {ARCHS[arch]} ==========")
    paths, labels = collect_images(root, task)
    counts = {CLASS_NAMES[task][i]: int((labels == i).sum()) for i in (0, 1)}
    print(f"Found {len(paths)} images: {counts}")

    (p_tr, y_tr), (p_val, y_val), (p_te, y_te) = split(paths, labels)
    train_ds = make_ds(p_tr, y_tr, img_size, batch, shuffle=True)
    val_ds = make_ds(p_val, y_val, img_size, batch, shuffle=False)
    test_ds = make_ds(p_te, y_te, img_size, batch, shuffle=False)

    # class weights for imbalance
    n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
    class_weight = {0: len(y_tr) / (2 * n0), 1: len(y_tr) / (2 * n1)}

    model, base = build_model(img_size, weights, arch)
    stem = f"{task}_model" if tag is None else f"{task}_{tag}"
    ckpt = out_dir / f"{stem}.keras"
    t_start = time.time()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_auc", mode="max",
                                        save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
    ]

    # Phase 1: train the classifier head
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                   class_weight=class_weight, callbacks=callbacks)

    # Phase 2: fine-tune the top layers of the pretrained backbone
    histories = [h1]
    if ft_epochs > 0 and arch != "custom_cnn":
        base.trainable = True
        for layer in base.layers[:-ft_layers]:
            layer.trainable = False
        compile_model(model, 1e-5)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs,
                       class_weight=class_weight, callbacks=callbacks)
        histories.append(h2)

    model.save(ckpt)  # best weights restored by EarlyStopping
    train_s = time.time() - t_start

    # Evaluation on held-out test split
    probs = model.predict(test_ds, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    names = CLASS_NAMES[task]
    report = classification_report(y_te, preds, target_names=names, output_dict=True,
                                   zero_division=0)
    cm = confusion_matrix(y_te, preds)
    print(classification_report(y_te, preds, target_names=names, zero_division=0))

    test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    sample = next(iter(test_ds))[0][:1]
    model(sample, training=False)
    t0 = time.perf_counter()
    for _ in range(20):
        model(sample, training=False)
    infer_ms = (time.perf_counter() - t0) / 20 * 1000
    history = {}
    for h in histories:
        for k, v in h.history.items():
            if k != "learning_rate":
                history.setdefault(k, []).extend(float(x) for x in v)
    summary = {
        "task": task,
        "arch": arch,
        "arch_name": ARCHS[arch],
        "params": int(model.count_params()),
        "size_mb": round(ckpt.stat().st_size / 1e6, 2),
        "train_s": round(train_s, 1),
        "infer_ms": round(infer_ms, 2),
        "classes": names,
        "positive_class": names[1],
        "img_size": img_size,
        "counts": counts,
        "split_sizes": {"train": len(p_tr), "val": len(p_val), "test": len(p_te)},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "fine_tune_start": len(histories[0].history["loss"]) if len(histories) > 1 else None,
    }
    (out_dir / f"{stem.replace('_model', '')}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if tag is None:
        plot_history(histories, out_dir / f"{task}_training_curves.png")
        plot_confusion(cm, names, out_dir / f"{task}_confusion_matrix.png")
    print(f"Saved {ckpt} and metrics/plots in {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["eye", "yawn", "both"], default="both")
    ap.add_argument("--data-dir", default=None,
                    help="Dataset folder (default: Kaggle input or kagglehub download)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--img-size", type=int, default=None,
                    help="Square input size (default: eye 96, yawn 128)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--fine-tune-epochs", type=int, default=6)
    ap.add_argument("--fine-tune-layers", type=int, default=30)
    ap.add_argument("--arch", choices=list(ARCHS), default="mobilenetv2",
                    help="CNN backbone (the app uses whatever is saved as <task>_model.keras)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random init instead of ImageNet weights (for quick tests)")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    print("GPUs:", tf.config.list_physical_devices("GPU") or "none (CPU training)")

    root = get_dataset_root(args.data_dir)
    print("Dataset root:", root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["eye", "yawn"] if args.task == "both" else [args.task]
    for task in tasks:
        train_task(task, root, out_dir, args.img_size or DEFAULT_IMG_SIZE[task],
                   args.batch, args.epochs, args.fine_tune_epochs,
                   args.fine_tune_layers, None if args.no_pretrained else "imagenet", args.arch)


if __name__ == "__main__":
    main()
