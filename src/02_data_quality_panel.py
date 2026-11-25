from pathlib import Path
import polars as pl

RAW_DIR = Path("data_raw/fdic")
OUT_DIR = Path("data_processed")

def main():
    print("=== PART 2: DATA QUALITY AUDIT & POINT-IN-TIME PANEL CONSTRUCTION ===")
    df_fin = pl.read_parquet(OUT_DIR / "financials_snappy.parquet")
    df_fail = pl.read_parquet(RAW_DIR / "failures.parquet")
    df_inst = pl.read_parquet(RAW_DIR / "institutions.parquet")

    df_fin = df_fin.with_columns([
        pl.col("CERT").cast(pl.Int64),
        pl.col("REPDTE").cast(pl.Utf8).str.slice(0, 8).str.to_date("%Y%m%d", strict=False).alias("rep_date")
    ]).filter(pl.col("rep_date").is_not_null())

    df_fin = df_fin.with_columns([
        pl.col("rep_date").dt.year().alias("year"),
        ((pl.col("rep_date").dt.year() // 10) * 10).alias("decade")
    ])

    check_cols = [c for c in ["ASSET", "EQ", "ROA", "NIMY", "NCLNLS", "RBC1AAJ", "DEPUNINS"] if c in df_fin.columns]
    print("\n[1] MISSING RATE (%) BY DECADE:")
    miss_by_decade = (
        df_fin.group_by("decade")
        .agg([pl.len().alias("n_rows")] + [(pl.col(c).is_null().mean() * 100).round(2).alias(f"{c}_miss%") for c in check_cols])
        .sort("decade")
    )
    print(miss_by_decade)

    n_bad_asset = df_fin.filter(pl.col("ASSET").is_null() | (pl.col("ASSET") <= 0)).height
    n_neg_eq = df_fin.filter(pl.col("EQ") < 0).height
    print(f"\n[2] ANOMALY & OUTLIER AUDIT:")
    print(f"  - Rows with ASSET <= 0 or Null (dropped) : {n_bad_asset:,}")
    print(f"  - Rows with Negative Equity (EQ < 0)     : {n_neg_eq:,} (kept as distress signal)")
    df_fin = df_fin.filter(pl.col("ASSET") > 0)

    fail_clean = (
        df_fail.select([
            pl.col("CERT").cast(pl.Int64),
            pl.col("FAILDATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).fill_null(
                pl.col("FAILDATE").cast(pl.Utf8).str.to_date("%m/%d/%Y", strict=False)
            ).alias("fail_date"),
            pl.col("COST").cast(pl.Float64, strict=False).alias("fail_cost"),
            pl.col("QBFASSET").cast(pl.Float64, strict=False).alias("fail_qbfasset"),
        ])
        .filter(pl.col("CERT").is_not_null() & pl.col("fail_date").is_not_null())
        .unique(subset=["CERT"], keep="first")
    )

    inst_cols = ["CERT"] + [c for c in ["ENDEFYMD", "CHANGEC1", "ACTIVE"] if c in df_inst.columns]
    inst_clean = df_inst.select(inst_cols).with_columns(pl.col("CERT").cast(pl.Int64)).unique(subset=["CERT"], keep="first")
    if "ENDEFYMD" in inst_clean.columns:
        inst_clean = inst_clean.with_columns(
            pl.col("ENDEFYMD").cast(pl.Utf8).str.slice(0, 8).str.to_date("%Y%m%d", strict=False).alias("end_date")
        )
    else:
        inst_clean = inst_clean.with_columns(pl.lit(None).cast(pl.Date).alias("end_date"))

    panel = df_fin.join(fail_clean, on="CERT", how="left").join(inst_clean.select(["CERT", "end_date"]), on="CERT", how="left")

    before_exit_filter = panel.height
    panel = panel.filter(
        (pl.col("fail_date").is_null() | (pl.col("rep_date") <= pl.col("fail_date"))) &
        (pl.col("end_date").is_null() | (pl.col("rep_date") <= pl.col("end_date")))
    )
    print(f"\n[3] EXIT & MERGER FILTERING:")
    print(f"  - Post-exit / post-merger phantom rows removed : {before_exit_filter - panel.height:,}")

    panel = panel.with_columns(
        (pl.col("fail_date") - pl.col("rep_date")).dt.total_days().alias("days_to_fail")
    ).with_columns(
        pl.when((pl.col("days_to_fail") > 0) & (pl.col("days_to_fail") <= 366))
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("target_fail_4q")
    )

    panel_clean = panel.filter(pl.col("rep_date") <= pl.date(2025, 6, 30))

    dev_df = panel_clean.filter(pl.col("rep_date") <= pl.date(2006, 12, 31))
    gap_df = panel_clean.filter((pl.col("rep_date") >= pl.date(2007, 1, 1)) & (pl.col("rep_date") <= pl.date(2007, 12, 31)))
    test_df = panel_clean.filter(pl.col("rep_date") >= pl.date(2008, 1, 1))

    print(f"\n[4] PANEL & SPLIT SUMMARY:")
    print(f"  - Total Valid Panel (<= 2025Q2) : {panel_clean.height:,} rows | Positive labels: {panel_clean['target_fail_4q'].sum():,}")
    print(f"  - Development (1984Q1 - 2006Q4) : {dev_df.height:,} rows | Positive labels: {dev_df['target_fail_4q'].sum():,} | Unique failed banks: {dev_df.filter(pl.col('target_fail_4q')==1)['CERT'].n_unique()}")
    print(f"  - Gap Period  (2007Q1 - 2007Q4) : {gap_df.height:,} rows | Positive labels: {gap_df['target_fail_4q'].sum():,}")
    print(f"  - Test Set    (2008Q1 - 2025Q2) : {test_df.height:,} rows | Positive labels: {test_df['target_fail_4q'].sum():,} | Unique failed banks: {test_df.filter(pl.col('target_fail_4q')==1)['CERT'].n_unique()}")

    panel_clean.write_parquet(OUT_DIR / "panel_clean.parquet", compression="snappy")
    print("\n[DONE] Đã lưu data_processed/panel_clean.parquet!")

if __name__ == "__main__":
    main()
