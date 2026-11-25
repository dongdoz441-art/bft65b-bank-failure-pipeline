from pathlib import Path
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml.classification import LogisticRegression as SparkLR
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

OUT_DIR = Path("data_processed")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        self.lower_bounds_ = np.nanquantile(X, self.lower_q, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

def run_spark_mllib_demo(X_dev, y_dev, X_test, y_test):
    spark = SparkSession.builder.appName("BFT65B-SparkMLlib").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    n_sub = min(150000, len(y_dev))
    cols = [f"f{i}" for i in range(X_dev.shape[1])]
    df_tr = pd.DataFrame(X_dev[-n_sub:], columns=cols); df_tr["label"] = y_dev[-n_sub:].astype(float)
    df_te = pd.DataFrame(X_test, columns=cols); df_te["label"] = y_test.astype(float)

    sdf_tr = spark.createDataFrame(df_tr)
    sdf_te = spark.createDataFrame(df_te)

    vec = VectorAssembler(inputCols=cols, outputCol="features")
    sdf_tr = vec.transform(sdf_tr).select("features", "label")
    sdf_te = vec.transform(sdf_te).select("features", "label")

    lr = SparkLR(maxIter=20, regParam=0.1)
    model = lr.fit(sdf_tr)
    preds = model.transform(sdf_te)

    eval_pr = BinaryClassificationEvaluator(metricName="areaUnderPR")
    eval_roc = BinaryClassificationEvaluator(metricName="areaUnderROC")
    pr_auc = eval_pr.evaluate(preds)
    roc_auc = eval_roc.evaluate(preds)
    spark.stop()
    return pr_auc, roc_auc

def main():
    print("=== PART 5: MODELING, COST-SENSITIVE THRESHOLDING & TEST EVALUATION ===")
    data = joblib.load(OUT_DIR / "ml_ready_data.joblib")
    X_dev, y_dev = data["X_dev"], data["y_dev"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_cols = data["feature_cols"]

    tscv = TimeSeriesSplit(n_splits=5, gap=40000)
    oof_lr = np.zeros(len(y_dev))
    oof_lgb = np.zeros(len(y_dev))
    oof_mask = np.zeros(len(y_dev), dtype=bool)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_dev), 1):
        lr = LogisticRegression(max_iter=500, class_weight="balanced", C=0.1, random_state=42)
        lr.fit(X_dev[tr_idx], y_dev[tr_idx])
        oof_lr[val_idx] = lr.predict_proba(X_dev[val_idx])[:, 1]

        clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )
        clf.fit(X_dev[tr_idx], y_dev[tr_idx])
        oof_lgb[val_idx] = clf.predict_proba(X_dev[val_idx])[:, 1]
        oof_mask[val_idx] = True

    pr_dev_lr = average_precision_score(y_dev[oof_mask], oof_lr[oof_mask])
    roc_dev_lr = roc_auc_score(y_dev[oof_mask], oof_lr[oof_mask])
    pr_dev_lgb = average_precision_score(y_dev[oof_mask], oof_lgb[oof_mask])
    roc_dev_lgb = roc_auc_score(y_dev[oof_mask], oof_lgb[oof_mask])

    print(f"\n[1] DEVELOPMENT TIME-SERIES CV RESULTS:")
    print(f"  - Logistic Regression : PR-AUC = {pr_dev_lr:.4f} | ROC-AUC = {roc_dev_lr:.4f}")
    print(f"  - LightGBM            : PR-AUC = {pr_dev_lgb:.4f} | ROC-AUC = {roc_dev_lgb:.4f}")

    valid_cost = (~np.isnan(data["dev_fail_cost"])) & (~np.isnan(data["dev_fail_asset"])) & (data["dev_fail_asset"] > 0)
    loss_severity = np.nanmedian(data["dev_fail_cost"][valid_cost] / data["dev_fail_asset"][valid_cost])
    if np.isnan(loss_severity) or loss_severity <= 0:
        loss_severity = 0.20
    c_fn = float(loss_severity)
    c_fp = 0.005
    tau_theory = c_fp / (c_fp + c_fn)

    thresholds = np.linspace(0.01, 0.90, 90)
    losses = []
    for t in thresholds:
        preds = (oof_lgb[oof_mask] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_dev[oof_mask], preds).ravel()
        losses.append(fn * c_fn + fp * c_fp)
    best_tau = float(thresholds[int(np.argmin(losses))])

    print(f"\n[2] COST-SENSITIVE THRESHOLD TUNING (On Development):")
    print(f"  - Median FDIC Loss/Asset Severity (C_FN) : {c_fn:.4f} | Inspection Cost (C_FP) : {c_fp:.4f}")
    print(f"  - Theoretical Optimal Threshold (tau*)   : {tau_theory:.4f}")
    print(f"  - Empirical OOF Optimal Threshold        : {best_tau:.4f}")

    final_lr = LogisticRegression(max_iter=500, class_weight="balanced", C=0.1, random_state=42).fit(X_dev, y_dev)
    final_lgb = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
    ).fit(X_dev, y_dev)

    prob_test_lr = final_lr.predict_proba(X_test)[:, 1]
    prob_test_lgb = final_lgb.predict_proba(X_test)[:, 1]
    pr_spark, roc_spark = run_spark_mllib_demo(X_dev, y_dev, X_test, y_test)

    print(f"\n[3] FINAL OUT-OF-TIME TEST SET EVALUATION (2008Q1 - 2025Q2):")
    print(f"  - Logistic Regression : PR-AUC = {average_precision_score(y_test, prob_test_lr):.4f} | ROC-AUC = {roc_auc_score(y_test, prob_test_lr):.4f}")
    print(f"  - LightGBM            : PR-AUC = {average_precision_score(y_test, prob_test_lgb):.4f} | ROC-AUC = {roc_auc_score(y_test, prob_test_lgb):.4f}")
    print(f"  - Spark MLlib (LR)    : PR-AUC = {pr_spark:.4f} | ROC-AUC = {roc_spark:.4f}")

    for name, tau in [("Default (tau=0.50)", 0.50), (f"Cost-Optimal (tau={tau_theory:.3f})", tau_theory)]:
        y_hat = (prob_test_lgb >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
        tot_loss = fn * c_fn + fp * c_fp
        recall = tp / (tp + fn)
        print(f"  -> {name:<26}: TP={tp}, FN={fn}, FP={fp} | Recall={recall*100:.1f}% | Total Loss={tot_loss:.2f}")

    prec_lgb, rec_lgb, _ = precision_recall_curve(y_test, prob_test_lgb)
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, prob_test_lr)
    plt.figure(figsize=(7, 5))
    plt.plot(rec_lgb, prec_lgb, label=f"LightGBM (PR-AUC={average_precision_score(y_test, prob_test_lgb):.3f})")
    plt.plot(rec_lr, prec_lr, linestyle="--", label=f"Logistic Reg (PR-AUC={average_precision_score(y_test, prob_test_lr):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Out-of-Time Test Precision-Recall Curve (2008Q1-2025Q2)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(FIG_DIR / "pr_curve_test.png", dpi=200)

    importances = final_lgb.feature_importances_
    active_cols = [c for c in feature_cols if c not in ("camels_m_eff_ratio", "camels_l_uninsured_dep")][:len(importances)]
    top_idx = np.argsort(importances)[-12:]
    plt.figure(figsize=(8, 5))
    plt.barh([active_cols[i] for i in top_idx], importances[top_idx], color="#2b5c8f")
    plt.xlabel("Split Importance"); plt.title("Top 12 CAMELS Predictors of Bank Failure (LightGBM)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=200)
    print("\n[DONE] Đã lưu biểu đồ vào figures/pr_curve_test.png và figures/feature_importance.png!")

if __name__ == "__main__":
    main()
