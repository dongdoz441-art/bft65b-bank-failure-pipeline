# BFT65B: U.S. Bank Failure Early-Warning Big-Data Pipeline (1984Q1–2025Q2)

End-to-end, leak-free, cost-sensitive machine learning pipeline built on FDIC BankFind Suite historical Call Reports (`1,682,739` bank-quarter observations).

## Repository Structure
- `fdic_download_starter.py`: Paginated FDIC API data downloader (`1984Q1–2026Q2`).
- `src/01_ingest_storage.py`: Schema harmonization, primary key integrity checks, and CSV vs. Parquet (Snappy/Zstd) I/O benchmark.
- `src/02_data_quality_panel.py`: Decade-level missingness audit, exit/merger filtering (`5,628` phantom rows removed), and forward 4-quarter failure labeling (`<= 2025Q2`).
- `src/03_polars_vs_spark.py`: Rolling window & lag computation benchmark (`Polars` vs. `Pandas` vs. `PySpark`) and physical execution plan inspection.
- `src/04_feature_pipeline.py`: CAMELS financial ratios + leak-free `scikit-learn` preprocessing pipeline (`fit` exclusively on `1984Q1–2006Q4`).
- `src/05_modeling_eval.py`: Purged `TimeSeriesSplit` CV, LightGBM vs. Logistic Regression vs. Spark MLlib evaluation, and Bayes cost-optimal thresholding.
- `run_all.py`: Master script to reproduce all results and figures end-to-end.

## How to Reproduce
1. Install dependencies: `pip install -r requirements.txt`
2. Download raw FDIC data into `data_raw/fdic/`: `python fdic_download_starter.py`
3. Run the entire pipeline: `python run_all.py`

## Citations & AI Disclosure
- **Data:** FDIC BankFind Suite API (`https://banks.data.fdic.gov/api/`).
- **AI Assistance:** Google Gemini was used for code structuring, Spark environment troubleshooting, and report editing; all experiments were executed and verified by the author.
