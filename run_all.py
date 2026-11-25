import subprocess, sys, time
SCRIPTS = ["src/01_ingest_storage.py", "src/02_data_quality_panel.py", "src/03_polars_vs_spark.py", "src/04_feature_pipeline.py", "src/05_modeling_eval.py"]
for i, s in enumerate(SCRIPTS, 1):
    print(f"\n[{i}/{len(SCRIPTS)}] Running {s}...")
    res = subprocess.run([sys.executable, s])
    if res.returncode != 0:
        sys.exit(res.returncode)
print("\nAll pipeline stages completed successfully!")
