"""Run on Kaggle GPU: python train.py --data DATASET_ROOT --output /kaggle/working/outputs"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from preprocess import load_config, transform


SEED = 75


def build_small_cnn(input_shape):
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.0, 0.06, fill_mode="constant"),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(1, activation="sigmoid"),
    ], name="small_cnn")


def build_deeper_cnn(input_shape):
    layers = tf.keras.layers
    reg = tf.keras.regularizers.l2(1e-4)
    return tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom((-0.05, 0.05), (-0.08, 0.08), fill_mode="constant"),
        layers.Conv2D(24, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Conv2D(24, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.MaxPooling2D(),
        layers.Conv2D(48, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Conv2D(48, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.MaxPooling2D(),
        layers.Conv2D(96, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.45),
        layers.Dense(1, activation="sigmoid"),
    ], name="deeper_cnn")


def build_sep_cnn(input_shape):
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.RandomTranslation(0.0, 0.08, fill_mode="constant"),
        layers.SeparableConv2D(32, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.SeparableConv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(48, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ], name="separable_cnn")


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.AUC(name="pr_auc", curve="PR")],
    )


def class_weight_for(y):
    counts = np.bincount(y.astype(int), minlength=2).astype(np.float64)
    total = counts.sum()
    return {0: float(total / (2 * counts[0])), 1: float(total / (2 * counts[1]))}


def threshold_from_validation(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def evaluate(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores, labels=[0, 1])),
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def fit_calibrators(y_val, p_val):
    clipped = np.clip(p_val, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(logit, y_val)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(p_val, y_val)
    return platt, isotonic


def apply_platt(model, p):
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logit)[:, 1]


def calibration_payload(y_true, scores):
    frac, means = calibration_curve(y_true, scores, n_bins=8, strategy="quantile")
    return {"mean_score": means.tolist(), "observed_fraction": frac.tolist()}


def convert_tflite(model_path, output_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    output_path.write_bytes(converter.convert())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise RuntimeError("Completed run already exists. Use a new output directory.")

    tf.keras.utils.set_random_seed(SEED)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("Enable a GPU accelerator in Kaggle before training.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    cfg = load_config(args.data / "preprocessing.json")
    df = pd.read_csv(args.data / "manifest.csv")
    assert df.uuid.is_unique
    assert df.groupby("sha256")["split"].nunique().max() == 1
    assert set(df["split"]) == {"train", "validation", "test"}

    arrays = []
    failures = []
    for i, row in df.iterrows():
        try:
            arrays.append(transform(args.data / row["path"], cfg))
        except Exception as exc:
            failures.append({"uuid": row.uuid, "error": str(exc)})
        if (i + 1) % 200 == 0:
            print(f"Preprocessed {i + 1}/{len(df)}", flush=True)
    if failures:
        (out / "preprocessing_failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError("Audio failures found; inspect report before changing the cohort.")
    x = np.stack(arrays)
    y = df.label.to_numpy(dtype=np.int32)
    train = df["split"].eq("train").to_numpy()
    val = df["split"].eq("validation").to_numpy()
    test = df["split"].eq("test").to_numpy()
    for mask in (train, val, test):
        assert set(y[mask]) == {0, 1}

    candidates = {"small_cnn": build_small_cnn, "deeper_cnn": build_deeper_cnn, "separable_cnn": build_sep_cnn}
    comparison = {}
    histories = {}
    predictions = {}
    weights = class_weight_for(y[train])

    for name, builder in candidates.items():
        model_dir = out / name
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.keras"
        model = builder(x.shape[1:])
        compile_model(model)
        history = model.fit(
            x[train],
            y[train],
            validation_data=(x[val], y[val]),
            epochs=args.epochs,
            batch_size=32,
            class_weight=weights,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_auc", mode="max", save_best_only=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=4, factor=0.5),
                tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
            ],
            verbose=2,
        )
        model = tf.keras.models.load_model(model_path)
        p_val = model.predict(x[val], verbose=0).ravel()
        p_test = model.predict(x[test], verbose=0).ravel()
        threshold = threshold_from_validation(y[val], p_val)

        platt, isotonic = fit_calibrators(y[val], p_val)
        p_val_platt = apply_platt(platt, p_val)
        p_test_platt = apply_platt(platt, p_test)
        platt_threshold = threshold_from_validation(y[val], p_val_platt)
        p_val_iso = isotonic.predict(p_val)
        p_test_iso = isotonic.predict(p_test)
        iso_threshold = threshold_from_validation(y[val], p_val_iso)

        comparison[name] = {
            "validation_raw": evaluate(y[val], p_val, threshold),
            "test_raw": evaluate(y[test], p_test, threshold),
            "test_platt": evaluate(y[test], p_test_platt, platt_threshold),
            "test_isotonic": evaluate(y[test], p_test_iso, iso_threshold),
        }
        histories[name] = history.history
        predictions[name] = {
            "test_raw": p_test,
            "test_platt": p_test_platt,
            "test_isotonic": p_test_iso,
            "threshold_platt": platt_threshold,
        }
        (model_dir / "platt.json").write_text(json.dumps({
            "coef": float(platt.coef_[0, 0]),
            "intercept": float(platt.intercept_[0]),
            "input": "logit(raw_score)",
        }, indent=2))
        (model_dir / "isotonic.json").write_text(json.dumps({
            "x_thresholds": isotonic.X_thresholds_.tolist(),
            "y_thresholds": isotonic.y_thresholds_.tolist(),
        }, indent=2))

    best_name = max(comparison, key=lambda n: comparison[n]["validation_raw"]["roc_auc"])
    shutil.copy2(out / best_name / "model.keras", out / "urti_model.keras")
    shutil.copy2(out / best_name / "platt.json", out / "platt_calibration.json")
    shutil.copy2(out / best_name / "isotonic.json", out / "isotonic_calibration.json")
    convert_tflite(out / "urti_model.keras", out / "urti_model.tflite")

    cfg["version"] = "urti-kaggle-multimodel-v2"
    cfg["threshold"] = predictions[best_name]["threshold_platt"]
    cfg["calibrated"] = True
    cfg["calibration"] = "platt"
    cfg["selected_model"] = best_name
    (out / "preprocessing.json").write_text(json.dumps(cfg, indent=2))

    p_best_raw = predictions[best_name]["test_raw"]
    p_best_platt = predictions[best_name]["test_platt"]
    p_best_iso = predictions[best_name]["test_isotonic"]
    metrics = {
        "selected_model": best_name,
        "selection_rule": "highest validation ROC-AUC; test set evaluated once after selection",
        "comparison": comparison,
        "selected_test_metrics": comparison[best_name]["test_platt"],
        "calibrated": True,
        "calibration": "platt fitted on validation split only",
        "calibration_curve": calibration_payload(y[test], p_best_platt),
        "train_size": int(train.sum()),
        "validation_size": int(val.sum()),
        "test_size": int(test.sum()),
        "class_weight": weights,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(histories, indent=2))

    result = df.loc[test, ["uuid", "label"]].copy()
    result["raw_score"] = p_best_raw
    result["platt_score"] = p_best_platt
    result["isotonic_score"] = p_best_iso
    result["prediction"] = (p_best_platt >= cfg["threshold"]).astype(int)
    result.to_csv(out / "test_predictions.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, history in histories.items():
        axes[0].plot(history["val_auc"], label=name)
    axes[0].set(title="Validation ROC-AUC", xlabel="Epoch")
    axes[0].legend()
    fpr, tpr, _ = roc_curve(y[test], p_best_platt)
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], "--")
    axes[1].set(title="Test ROC", xlabel="False positive rate", ylabel="Sensitivity")
    cal = metrics["calibration_curve"]
    axes[2].plot(cal["mean_score"], cal["observed_fraction"], "o-")
    axes[2].plot([0, 1], [0, 1], "--")
    axes[2].set(title="Test calibration", xlabel="Mean calibrated score", ylabel="Observed fraction")
    fig.tight_layout()
    fig.savefig(out / "evaluation.png")
    plt.close(fig)

    for name in ["manifest.csv", "dataset_summary.json"]:
        shutil.copy2(args.data / name, out / name)
    for name in ["preprocess.py", "predict.py", "train.py", "README.md"]:
        shutil.copy2(Path(__file__).parent / name, out / name)
    (out / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))

    from predict import predict
    first = df.loc[test].iloc[0]
    exported = predict(args.data / first["path"], out)["raw_score"]
    assert abs(exported - float(p_best_raw[0])) < 1e-5, "Exported inference mismatch"

    shutil.make_archive(str(out.parent / "urti_model_bundle"), "zip", out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
