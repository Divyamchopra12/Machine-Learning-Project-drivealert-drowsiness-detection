"""
Classical machine-learning models for the Driver Drowsiness Detection System,
compared with the MobileNetV2 CNN on the SAME Kaggle data and the SAME split.

Models (scikit-learn; XGBoost if installed):
  Logistic Regression · K-Nearest Neighbours · Support Vector Machine (RBF)
  Decision Tree · Random Forest · Gaussian Naive Bayes · Gradient Boosting
  AdaBoost · Extra Trees · MLP Neural Network (ANN) · Linear Discriminant Analysis · XGBoost

Features: HOG + intensity histogram (see features.py).

  python train_classical.py --task both          # eye (open/closed) and yawn models
  python train_classical.py --task eye --tune    # + GridSearchCV for RF and SVM
  python train_classical.py --data-dir <folder>  # dataset folder (default: Kaggle / kagglehub)

Outputs
  models/<task>_classical.joblib          best model (by F1) – usable live in the app
  models/<task>_classical_metrics.json    all results (shown on the About page)
  models/<task>_model_comparison.png      bar chart: accuracy / F1 per model (+ CNN)
  models/<task>_rf_importance.png         where Random Forest looks (HOG importance map)
  results/<task>_classical_comparison.md  table for the report
"""
import argparse
import warnings
import io
import json
import os
import pickle
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from features import FEAT_SIZE, HOG_LEN, batch_features

# --- same dataset handling and split as train_model.py (so results are comparable) ---
KAGGLE_DATASET = "serenaraju/yawn-eye-dataset-new"
KAGGLE_INPUT_DIR = "/kaggle/input/yawn-eye-dataset-new"
CLASS_MAP = {"eye": {"open": 0, "closed": 1}, "yawn": {"no_yawn": 0, "yawn": 1}}
CLASS_NAMES = {"eye": ["open", "closed"], "yawn": ["no_yawn", "yawn"]}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SEED = 42
warnings.filterwarnings("ignore", category=FutureWarning)   # sklearn SVC(probability=True) notice


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


def collect_images(root, task):
    names, paths, labels, seen = CLASS_MAP[task], [], [], set()
    for dirpath, _, files in os.walk(root):
        key = Path(dirpath).name.lower().replace("-", "_").replace(" ", "_")
        if key not in names:
            continue
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                full = os.path.join(dirpath, f)
                ident = (key, f, os.path.getsize(full))
                if ident not in seen:
                    seen.add(ident)
                    paths.append(full)
                    labels.append(names[key])
    if not paths:
        raise SystemExit(f"No images for task '{task}' under {root}")
    return np.array(paths), np.array(labels, dtype=np.int32)


def split(paths, labels):
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30, stratify=labels,
                                                random_state=SEED)
    p_val, p_te, y_val, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp,
                                                random_state=SEED)
    return (p_tr, y_tr), (p_val, y_val), (p_te, y_te)


def load_features(paths):
    imgs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        imgs.append(img if img is not None else np.zeros((FEAT_SIZE, FEAT_SIZE), np.uint8))
    return batch_features(imgs)


# ---------------------------------------------------------------- models ----
def build_models(n_features, n_samples):
    # PCA size must stay below the number of samples in each cross-validation fold
    pca = max(2, min(100, n_features, int(n_samples * 0.5)))
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, C=0.5, class_weight="balanced")),
        "K-Nearest Neighbours": make_pipeline(StandardScaler(), PCA(pca, random_state=SEED),
                                              KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "SVM (RBF kernel)": make_pipeline(StandardScaler(), PCA(pca, random_state=SEED), SVC(
            kernel="rbf", C=4, gamma="scale", probability=True, class_weight="balanced",
            random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(max_depth=14, min_samples_leaf=3,
                                                class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=1, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=SEED),
        "Naive Bayes": make_pipeline(StandardScaler(), PCA(pca, random_state=SEED), GaussianNB()),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                                            random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced",
                                            random_state=SEED),
        "MLP Neural Network (ANN)": make_pipeline(StandardScaler(), PCA(pca, random_state=SEED), MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=400, early_stopping=len_ok(n_samples),
            random_state=SEED)),
        "Linear Discriminant Analysis": make_pipeline(StandardScaler(), PCA(pca, random_state=SEED),
                                                      LinearDiscriminantAnalysis()),
    }
    try:  # optional: pip install xgboost
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.9,
                                          colsample_bytree=0.5, eval_metric="logloss", random_state=SEED)
    except ImportError:
        pass
    return models


def len_ok(n_samples):
    """MLP early stopping needs a validation slice, so only use it with enough data."""
    return n_samples >= 200


def tune(name, model, x, y):
    """Small GridSearchCV for the two models examiners ask about most."""
    if name == "Random Forest":
        grid = {"n_estimators": [200, 400], "max_depth": [None, 20], "min_samples_leaf": [1, 3]}
    elif name.startswith("SVM"):
        grid = {"svc__C": [1, 4, 10], "svc__gamma": ["scale", 0.005]}
    else:
        return model, None
    gs = GridSearchCV(model, grid, cv=3, scoring="f1", n_jobs=-1)
    gs.fit(x, y)
    print(f"    best params: {gs.best_params_}")
    return gs.best_estimator_, gs.best_params_


def size_kb(model):
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return buf.tell() / 1024


# ----------------------------------------------------------------- plots ----
def plot_comparison(rows, task, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["model"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(11, len(rows) * 0.85), 4.8))
    ax.bar(x - 0.2, [r["accuracy"] * 100 for r in rows], 0.4, label="Accuracy %", color="#22D3EE")
    ax.bar(x + 0.2, [r["f1"] * 100 for r in rows], 0.4, label="F1 %", color="#A78BFA")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.set_title(f"{task} model comparison (held-out test set)")
    ax.legend()
    for i, r in enumerate(rows):
        ax.text(i + 0.2, r["f1"] * 100 + 1.5, f"{r['f1'] * 100:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_rf_importance(rf, out_png):
    """HOG layout: 7x7 blocks x (4 cells x 9 bins). Sum importance per block -> heat map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp = rf.feature_importances_[:HOG_LEN].reshape(7, 7, 36).sum(-1)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(imp.T, cmap="inferno")
    ax.set_title("Random Forest: where it looks")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# -------------------------------------------------------------- training ----
def run_task(task, root, out_dir, res_dir, do_tune, cv_folds, max_per_class):
    print(f"\n========== Classical models: '{task}' ==========")
    paths, labels = collect_images(root, task)
    (p_tr, y_tr), (p_val, y_val), (p_te, y_te) = split(paths, labels)
    # classical models have no early stopping, so train on train + validation
    p_fit, y_fit = np.concatenate([p_tr, p_val]), np.concatenate([y_tr, y_val])
    if max_per_class:
        rng = np.random.default_rng(SEED)
        keep = np.concatenate([rng.permutation(np.where(y_fit == c)[0])[:max_per_class] for c in (0, 1)])
        p_fit, y_fit = p_fit[keep], y_fit[keep]
    print(f"Extracting HOG features: {len(p_fit)} train / {len(p_te)} test images ...")
    t0 = time.time()
    x_fit, x_te = load_features(p_fit), load_features(p_te)
    print(f"  {x_fit.shape[1]} features per image ({time.time() - t0:.1f}s)")

    names = CLASS_NAMES[task]
    rows, fitted, cms = [], {}, {}
    for name, model in build_models(x_fit.shape[1], len(x_fit)).items():
        print(f"  - {name}")
        params = None
        cv_mean = cv_std = None
        if cv_folds:
            cv = StratifiedKFold(cv_folds, shuffle=True, random_state=SEED)
            sc = cross_val_score(model, x_fit, y_fit, cv=cv, scoring="accuracy", n_jobs=-1)
            cv_mean, cv_std = float(sc.mean()), float(sc.std())
        if do_tune:
            model, params = tune(name, model, x_fit, y_fit)
        t0 = time.time()
        model.fit(x_fit, y_fit)
        train_s = time.time() - t0
        t0 = time.time()
        prob = model.predict_proba(x_te)[:, 1]
        infer_ms = (time.time() - t0) / len(x_te) * 1000
        pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(y_te, pred, labels=[0, 1])
        row = dict(model=name, family="classical",
                   accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred, zero_division=0),
                   recall=recall_score(y_te, pred, zero_division=0), f1=f1_score(y_te, pred, zero_division=0),
                   auc=roc_auc_score(y_te, prob) if len(set(y_te)) > 1 else None,
                   cv_accuracy=cv_mean, cv_std=cv_std, train_s=train_s, infer_ms=infer_ms,
                   size_kb=size_kb(model), params=params)
        rows.append({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in row.items()})
        fitted[name], cms[name] = model, cm.tolist()
        print(f"      acc {row['accuracy'] * 100:.1f}%  F1 {row['f1']:.3f}  "
              f"train {train_s:.1f}s  {infer_ms:.2f} ms/img")

    # add every trained CNN architecture for comparison (train_dl_compare.py)
    dl_file = out_dir / f"{task}_dl_metrics.json"
    if dl_file.exists():
        for r in json.loads(dl_file.read_text(encoding="utf-8"))["results"]:
            rows.append(dict(model=r["model"], family="deep learning", accuracy=r["accuracy"],
                             precision=r["precision"], recall=r["recall"], f1=r["f1"], auc=r["auc"],
                             cv_accuracy=None, cv_std=None, train_s=r["train_s"], infer_ms=r["infer_ms"],
                             size_kb=r["size_kb"], params=None))
            cms[r["model"]] = r.get("confusion_matrix")
    # add the app's main CNN for comparison (from train_model.py results)
    cnn_file = out_dir / f"{task}_metrics.json"
    if cnn_file.exists() and not dl_file.exists():
        m = json.loads(cnn_file.read_text(encoding="utf-8"))
        rep = m.get("classification_report", {}).get(names[1], {})
        tm = m.get("test_metrics", {})
        cnn_cm = m.get("confusion_matrix")
        rows.append(dict(model="MobileNetV2 CNN", family="deep learning",
                         accuracy=m.get("classification_report", {}).get("accuracy", tm.get("accuracy")),
                         precision=rep.get("precision", tm.get("precision")), recall=rep.get("recall", tm.get("recall")),
                         f1=rep.get("f1-score"), auc=tm.get("auc"), cv_accuracy=None, cv_std=None,
                         train_s=None, infer_ms=None,
                         size_kb=(out_dir / f"{task}_model.keras").stat().st_size / 1024
                         if (out_dir / f"{task}_model.keras").exists() else None, params=None))
        if cnn_cm:
            cms["MobileNetV2 CNN"] = cnn_cm
        if rows[-1]["f1"] is None:
            p, r = rows[-1]["precision"] or 0, rows[-1]["recall"] or 0
            rows[-1]["f1"] = 2 * p * r / (p + r) if p + r else 0.0

    classical = [r for r in rows if r["family"] == "classical"]
    best = max(classical, key=lambda r: (r["f1"], r["accuracy"]))
    joblib.dump({"model": fitted[best["model"]], "name": best["model"], "task": task,
                 "classes": names, "feature": "hog64+hist16"}, out_dir / f"{task}_classical.joblib")
    print(f"\n  Best classical model: {best['model']} (F1 {best['f1']:.3f}) -> {task}_classical.joblib")

    summary = {"task": task, "classes": names, "best_classical": best["model"],
               "features": f"HOG {FEAT_SIZE}x{FEAT_SIZE} ({HOG_LEN}) + 16-bin histogram",
               "train_size": int(len(y_fit)), "test_size": int(len(y_te)),
               "results": rows, "confusion_matrices": cms}
    (out_dir / f"{task}_classical_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_comparison(rows, task, out_dir / f"{task}_model_comparison.png")
    if "Random Forest" in fitted:
        plot_rf_importance(fitted["Random Forest"], out_dir / f"{task}_rf_importance.png")

    def pct(v):
        return "—" if v is None else f"{v * 100:.1f}%"

    def num(v, digits):
        return "—" if v is None else f"{v:.{digits}f}"

    md = [f"# {task} – classical ML vs deep learning", "",
          f"Features: {summary['features']} · train {len(y_fit)} / test {len(y_te)} images", "",
          "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV acc | Train (s) | ms / image | Size (KB) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -(r["f1"] or 0)):
        cv = "—" if r["cv_accuracy"] is None else f"{r['cv_accuracy'] * 100:.1f} ± {r['cv_std'] * 100:.1f}"
        star = " ⭐" if r["model"] == best["model"] else ""
        md.append(f"| {r['model']}{star} | {pct(r['accuracy'])} | {pct(r['precision'])} | "
                  f"{pct(r['recall'])} | {num(r['f1'], 3)} | {num(r['auc'], 3)} | {cv} | "
                  f"{num(r['train_s'], 1)} | {num(r['infer_ms'], 2)} | {num(r['size_kb'], 0)} |")
    (res_dir / f"{task}_classical_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["eye", "yawn", "both"], default="both")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--tune", action="store_true", help="GridSearchCV for Random Forest and SVM")
    ap.add_argument("--cv", type=int, default=5, help="k-fold cross-validation folds (0 = skip)")
    ap.add_argument("--max-per-class", type=int, default=None, help="cap training images per class (speed)")
    args = ap.parse_args()

    root = get_dataset_root(args.data_dir)
    out_dir, res_dir = Path(args.out), Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    for task in (["eye", "yawn"] if args.task == "both" else [args.task]):
        run_task(task, root, out_dir, res_dir, args.tune, args.cv, args.max_per_class)


if __name__ == "__main__":
    main()
