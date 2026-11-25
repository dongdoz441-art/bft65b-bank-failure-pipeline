import time
import tracemalloc
from pathlib import Path
import pandas as pd
import polars as pl
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

OUT_DIR = Path("data_processed")

def benchmark_pandas(path: Path):
    tracemalloc.start()
    t0 = time.perf_counter()

    df = pd.read_parquet(path, columns=["CERT", "rep_date", "ASSET", "EQ", "ROA", "NCLNLS", "LNLSGR"])
    df = df.sort_values(["CERT", "rep_date"]).reset_index(drop=True)
    df["eq_ratio"] = df["EQ"] / df["ASSET"]
    df["npl_ratio"] = df["NCLNLS"] / df["LNLSGR"].replace(0, float("nan"))

    g = df.groupby("CERT", sort=False)
    df["roa_roll4_mean"] = g["ROA"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["roa_roll4_std"] = g["ROA"].transform(lambda s: s.rolling(4, min_periods=2).std())
    df["eq_ratio_lag1"] = g["eq_ratio"].shift(1)
    df["eq_ratio_lag4"] = g["eq_ratio"].shift(4)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return df, elapsed, peak_bytes / (1024 * 1024)

def benchmark_polars(path: Path):
    tracemalloc.start()
    t0 = time.perf_counter()

    q = (
        pl.scan_parquet(path)
        .select(["CERT", "rep_date", "ASSET", "EQ", "ROA", "NCLNLS", "LNLSGR"])
        .sort(["CERT", "rep_date"])
        .with_columns([
            (pl.col("EQ") / pl.col("ASSET")).alias("eq_ratio"),
            pl.when(pl.col("LNLSGR") != 0).then(pl.col("NCLNLS") / pl.col("LNLSGR")).otherwise(None).alias("npl_ratio"),
        ])
        .with_columns([
            pl.col("ROA").rolling_mean(window_size=4, min_samples=1).over("CERT").alias("roa_roll4_mean"),
            pl.col("ROA").rolling_std(window_size=4, min_samples=2).over("CERT").alias("roa_roll4_std"),
            pl.col("eq_ratio").shift(1).over("CERT").alias("eq_ratio_lag1"),
            pl.col("eq_ratio").shift(4).over("CERT").alias("eq_ratio_lag4"),
        ])
    )
    df = q.collect()

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return df, elapsed, peak_bytes / (1024 * 1024)

def benchmark_pyspark(path: Path):
    spark = (
        SparkSession.builder.appName("BFT65B-BankFailure-Benchmark")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    t0 = time.perf_counter()
    sdf = spark.read.parquet(str(path)).select("CERT", "rep_date", "ASSET", "EQ", "ROA", "NCLNLS", "LNLSGR")
    sdf = sdf.withColumn("eq_ratio", F.col("EQ") / F.col("ASSET")).withColumn(
        "npl_ratio", F.when(F.col("LNLSGR") != 0, F.col("NCLNLS") / F.col("LNLSGR")).otherwise(None)
    )

    w_order = Window.partitionBy("CERT").orderBy("rep_date")
    w_roll4 = Window.partitionBy("CERT").orderBy("rep_date").rowsBetween(-3, 0)

    sdf_out = (
        sdf.withColumn("roa_roll4_mean", F.avg("ROA").over(w_roll4))
        .withColumn("roa_roll4_std", F.stddev_samp("ROA").over(w_roll4))
        .withColumn("eq_ratio_lag1", F.lag("eq_ratio", 1).over(w_order))
        .withColumn("eq_ratio_lag4", F.lag("eq_ratio", 4).over(w_order))
    )

    print("\n[3] PYSPARK PHYSICAL EXECUTION PLAN (df.explain()):")
    sdf_out.explain(mode="formatted")

    count_rows = sdf_out.count()
    elapsed = time.perf_counter() - t0

    sample_spark = sdf_out.filter(F.col("CERT") == 3511).orderBy("rep_date").toPandas()
    spark.stop()
    return sample_spark, elapsed, count_rows

def main():
    print("=== PART 3: SCALABLE PROCESSING — POLARS VS PANDAS VS PYSPARK ===")
    panel_path = OUT_DIR / "panel_clean.parquet"

    df_pd, t_pd, ram_pd = benchmark_pandas(panel_path)
    df_pl, t_pl, ram_pl = benchmark_polars(panel_path)
    sample_sp, t_sp, sp_rows = benchmark_pyspark(panel_path)

    print(f"\n[1] BENCHMARK TABLE:")
    print(f"{'Engine':<15} | {'Wall Time (s)':>13} | {'Speedup vs Pandas':>17} | {'Python Peak RAM (MB)':>20}")
    print("-" * 74)
    print(f"{'Pandas':<15} | {t_pd:>13.3f} | {'1.00x':>17} | {ram_pd:>20.2f}")
    print(f"{'Polars (Lazy)':<15} | {t_pl:>13.3f} | {t_pd/t_pl:>16.2f}x | {ram_pl:>20.2f}")
    print(f"{'PySpark (Local)':<15} | {t_sp:>13.3f} | {t_pd/t_sp:>16.2f}x | {'JVM Managed (4GB)':>20}")

    sample_pl = df_pl.filter(pl.col("CERT") == 3511).sort("rep_date").to_pandas()
    diff_roll = (sample_pl["roa_roll4_mean"] - sample_sp["roa_roll4_mean"]).abs().max()
    diff_lag = (sample_pl["eq_ratio_lag1"] - sample_sp["eq_ratio_lag1"]).abs().max()
    print(f"\n[2] POLARS vs PYSPARK NUMERICAL EQUIVALENCE CHECK (CERT=3511):")
    print(f"  - Max absolute difference (roa_roll4_mean) : {diff_roll:.12f}")
    print(f"  - Max absolute difference (eq_ratio_lag1)  : {diff_lag:.12f}")
    print("\n[DONE] Part 3 hoàn tất!")

if __name__ == "__main__":
    main()
