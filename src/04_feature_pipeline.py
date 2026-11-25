from pathlib import Path
import joblib
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

OUT_DIR = Path("data_processed")

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

def build_camels_features(df: pl.DataFrame) -> pl.DataFrame:
    def safe_div(num, den):
        return pl.when(pl.col(den).is_not_null() & (pl.col(den) != 0)).then(pl.col(num) / pl.col(den)).otherwise(None)

    df = df.sort(["CERT", "rep_date"]).with_columns([
        safe_div("EQ", "ASSET").alias("camels_c_eq_asset"),
        pl.col("RBC1AAJ").cast(pl.Float64, strict=False).alias("camels_c_tier1_rbc") if "RBC1AAJ" in df.columns else pl.lit(None).cast(pl.Float64).alias("camels_c_tier1_rbc"),
        safe_div("NCLNLS", "LNLSGR").alias("camels_a_npl_ratio"),
        safe_div("LNATRES", "NCLNLS").alias("camels_a_reserve_cov"),
        safe_div("NTLNLS", "LNLSGR").alias("camels_a_net_chargeoff"),
        pl.col("EEFFR").cast(pl.Float64, strict=False).alias("camels_m_eff_ratio") if "EEFFR" in df.columns else pl.lit(None).cast(pl.Float64).alias("camels_m_eff_ratio"),
        safe_div("NONIX", "ASSET").alias("camels_m_nonint_exp"),
        pl.col("ROA").cast(pl.Float64, strict=False).alias("camels_e_roa"),
        pl.col("ROE").cast(pl.Float64, strict=False).alias("camels_e_roe") if "ROE" in df.columns else pl.lit(None).cast(pl.Float64).alias("camels_e_roe"),
        pl.col("NIMY").cast(pl.Float64, strict=False).alias("camels_e_nim") if "NIMY" in df.columns else pl.lit(None).cast(pl.Float64).alias("camels_e_nim"),
        safe_div("CHBAL", "ASSET").alias("camels_l_cash_asset"),
        safe_div("LNLSGR", "DEP").alias("camels_l_loan_to_dep"),
        safe_div("DEPUNINS", "DEP").alias("camels_l_uninsured_dep") if "DEPUNINS" in df.columns else pl.lit(None).cast(pl.Float64).alias("camels_l_uninsured_dep"),
        pl.col("ASSET").log().alias("camels_s_log_asset"),
    ])

    df = df.with_columns([
        (pl.col("camels_c_eq_asset") - pl.col("camels_c_eq_asset").shift(1).over("CERT")).alias("dyn_eq_qoq"),
        (pl.col("camels_c_eq_asset") - pl.col("camels_c_eq_asset").shift(4).over("CERT")).alias("dyn_eq_yoy"),
        (pl.col("camels_a_npl_ratio") - pl.col("camels_a_npl_ratio").shift(4).over("CERT")).alias("dyn_npl_yoy"),
        (pl.col("camels_e_roa") - pl.col("camels_e_roa").shift(4).over("CERT")).alias("dyn_roa_yoy"),
        pl.col("camels_e_roa").rolling_std(window_size=4, min_samples=2).over("CERT").alias("dyn_roa_vol4q"),
        (pl.col("camels_s_log_asset") - pl.col("camels_s_log_asset").shift(4).over("CERT")).alias("dyn_asset_growth_yoy"),
    ])
    return df

def main():
    print("=== PART 4: LEAK-FREE CAMELS FEATURE ENGINEERING PIPELINE ===")
    panel = pl.read_parquet(OUT_DIR / "panel_clean.parquet")
    df_feat = build_camels_features(panel)

    feature_cols = [c for c in df_feat.columns if c.startswith("camels_") or c.startswith("dyn_")]
    print(f"Total engineered CAMELS & dynamic features: {len(feature_cols)}")

    dev_df = df_feat.filter(pl.col("rep_date") <= pl.date(2006, 12, 31))
    test_df = df_feat.filter(pl.col("rep_date") >= pl.date(2008, 1, 1))

    X_dev = dev_df.select(feature_cols).to_numpy()
    y_dev = dev_df["target_fail_4q"].to_numpy()

    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df["target_fail_4q"].to_numpy()

    prep_pipeline = Pipeline([
        ("winsorizer", QuantileWinsorizer(lower_q=0.01, upper_q=0.99)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    X_dev_trans = prep_pipeline.fit_transform(X_dev)
    X_test_trans = prep_pipeline.transform(X_test)

    joblib.dump({
        "pipeline": prep_pipeline,
        "feature_cols": feature_cols,
        "X_dev": X_dev_trans,
        "y_dev": y_dev,
        "dev_dates": dev_df["rep_date"].to_numpy(),
        "dev_fail_cost": dev_df["fail_cost"].to_numpy(),
        "dev_fail_asset": dev_df["fail_qbfasset"].to_numpy(),
        "X_test": X_test_trans,
        "y_test": y_test,
        "test_fail_cost": test_df["fail_cost"].to_numpy(),
        "test_fail_asset": test_df["fail_qbfasset"].to_numpy(),
    }, OUT_DIR / "ml_ready_data.joblib")

    print(f"  - X_dev shape : {X_dev_trans.shape} (Positive rate: {y_dev.mean()*100:.3f}%)")
    print(f"  - X_test shape: {X_test_trans.shape} (Positive rate: {y_test.mean()*100:.3f}%)")
    print("[DONE] Part 4 hoàn tất!")

if __name__ == "__main__":
    main()
