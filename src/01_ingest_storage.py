import time
from pathlib import Path
import polars as pl

RAW_DIR = Path("data_raw/fdic")
OUT_DIR = Path("data_processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)

def main():
    print("=== PART 1: DATA INGESTION, INTEGRITY CHECK & STORAGE BENCHMARK ===")
    fin_dir = RAW_DIR / "financials"
    fail_path = RAW_DIR / "failures.parquet"
    inst_path = RAW_DIR / "institutions.parquet"

    # Đọc từng file quý và dùng diagonal_relaxed để tự động đồng nhất Int64 -> Float64
    pq_files = sorted(fin_dir.glob("*.parquet"))
    dfs = []
    for f in pq_files:
        d = pl.read_parquet(f)
        # Ép các cột số nguyên (trừ CERT) về Float64 để tránh SchemaError giữa các quý
        cast_exprs = [
            pl.col(c).cast(pl.Float64)
            for c, dt in zip(d.columns, d.dtypes)
            if c not in ("CERT", "REPDTE") and dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
        ]
        if cast_exprs:
            d = d.with_columns(cast_exprs)
        dfs.append(d)

    df_fin = pl.concat(dfs, how="diagonal_relaxed")
    df_fail = pl.read_parquet(fail_path)
    df_inst = pl.read_parquet(inst_path)

    print(f"\n[1] DATASET ROW COUNTS:")
    print(f"  - Financials   : {df_fin.height:,} rows x {df_fin.width} cols")
    print(f"  - Failures     : {df_fail.height:,} rows x {df_fail.width} cols")
    print(f"  - Institutions : {df_inst.height:,} rows x {df_inst.width} cols")

    n_unique_pk = df_fin.select(["CERT", "REPDTE"]).unique().height
    n_dups = df_fin.height - n_unique_pk
    print(f"\n[2] PRIMARY KEY INTEGRITY (CERT, REPDTE):")
    print(f"  - Unique pairs : {n_unique_pk:,} | Duplicates : {n_dups:,}")
    if n_dups > 0:
        df_fin = df_fin.unique(subset=["CERT", "REPDTE"], keep="last")

    pq_snappy = OUT_DIR / "financials_snappy.parquet"
    pq_zstd = OUT_DIR / "financials_zstd.parquet"
    csv_temp = OUT_DIR / "financials_temp.csv"

    df_fin.write_parquet(pq_snappy, compression="snappy")
    df_fin.write_parquet(pq_zstd, compression="zstd")
    df_fin.write_csv(csv_temp)

    cols_5 = [c for c in ["CERT", "REPDTE", "ASSET", "EQ", "ROA"] if c in df_fin.columns]

    t0 = time.perf_counter(); _ = pl.read_csv(csv_temp, infer_schema_length=10000); t_csv_full = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = pl.read_csv(csv_temp, columns=cols_5, infer_schema_length=10000); t_csv_5 = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = pl.read_parquet(pq_snappy); t_snappy_full = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = pl.read_parquet(pq_snappy, columns=cols_5); t_snappy_5 = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = pl.read_parquet(pq_zstd); t_zstd_full = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = pl.read_parquet(pq_zstd, columns=cols_5); t_zstd_5 = time.perf_counter() - t0

    s_csv = get_dir_size_mb(csv_temp)
    s_snappy = get_dir_size_mb(pq_snappy)
    s_zstd = get_dir_size_mb(pq_zstd)

    print(f"\n[3] STORAGE & I/O BENCHMARK TABLE:")
    print(f"{'Format':<18} | {'Size (MB)':>10} | {'Ratio':>8} | {'Full Scan (s)':>13} | {'5-Col Read (s)':>14}")
    print("-" * 74)
    print(f"{'Raw CSV':<18} | {s_csv:>10.2f} | {'1.00x':>8} | {t_csv_full:>13.3f} | {t_csv_5:>14.3f}")
    print(f"{'Parquet (Snappy)':<18} | {s_snappy:>10.2f} | {s_csv/s_snappy:>7.2f}x | {t_snappy_full:>13.3f} | {t_snappy_5:>14.3f}")
    print(f"{'Parquet (Zstd)':<18} | {s_zstd:>10.2f} | {s_csv/s_zstd:>7.2f}x | {t_zstd_full:>13.3f} | {t_zstd_5:>14.3f}")

    if csv_temp.exists():
        csv_temp.unlink()
    print("\n[DONE] Part 1 hoàn tất!")

if __name__ == "__main__":
    main()
