"""
PVDAQ — Download ALL System Metrics
=====================================
Downloads metrics parquet files for all available systems.
Metrics tell you what sensors each system has (power, temp, irradiance etc.)

REQUIREMENTS (run once):
    pip install boto3 --break-system-packages
    sudo apt install awscli  # if not already installed

HOW TO RUN:
    python3 download_all_metrics.py

OUTPUT:
    ~/Desktop/solar-data-all/metrics/
        metrics__system_4.parquet
        metrics__system_10.parquet
        ... (one per system)
    ~/Desktop/solar-data-all/metrics_summary.csv  ← human readable summary
"""

import os
import subprocess
import pandas as pd

HOME     = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME, "Desktop", "solar-data-all")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
BUCKET   = "oedi-data-lake"
PREFIX   = "pvdaq/parquet/metrics/"

os.makedirs(METRICS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Check requirements
# ─────────────────────────────────────────────
print("=" * 60)
print("Checking requirements...")
print("=" * 60)

# Check AWS CLI
result = subprocess.run(["aws", "--version"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ AWS CLI: {result.stdout.strip()}")
else:
    print("❌ AWS CLI not found! Run: sudo apt install awscli")
    exit(1)

# Check pyarrow for parquet reading
try:
    import pyarrow
    print(f"✅ pyarrow: {pyarrow.__version__}")
except ImportError:
    print("❌ pyarrow not found! Run: pip install pyarrow --break-system-packages")
    exit(1)

# ─────────────────────────────────────────────
# STEP 2: List all metrics files on S3
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("STEP 1: Listing all metrics files on S3...")
print(f"{'='*60}")

result = subprocess.run(
    ["aws", "s3", "ls", "--no-sign-request", f"s3://{BUCKET}/{PREFIX}"],
    capture_output=True, text=True
)

if result.returncode != 0:
    print(f"❌ Could not list S3: {result.stderr}")
    exit(1)

# Parse the listing
metrics_files = []
for line in result.stdout.strip().split("\n"):
    parts = line.split()
    if len(parts) >= 4 and parts[3].endswith(".parquet"):
        filename = parts[3]  # e.g. metrics__system_4__part000.parquet
        metrics_files.append(filename)

print(f"✅ Found {len(metrics_files)} metrics files on S3")

# ─────────────────────────────────────────────
# STEP 3: Download each metrics file
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("STEP 2: Downloading metrics files...")
print(f"{'='*60}")

downloaded = 0
skipped    = 0
errors     = 0

for i, filename in enumerate(metrics_files, 1):
    # Clean filename: metrics__system_4__part000.parquet → metrics__system_4.parquet
    sys_id = filename.replace("metrics__system_", "").replace("__part000.parquet", "")
    local_name = f"metrics__system_{sys_id}.parquet"
    local_path = os.path.join(METRICS_DIR, local_name)

    # Skip if already exists
    if os.path.isfile(local_path):
        skipped += 1
        if i % 20 == 0:
            print(f"  ⏭️  [{i}/{len(metrics_files)}] Skipping already downloaded files...")
        continue

    # Download
    s3_key = f"{PREFIX}{filename}"
    result = subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request",
         f"s3://{BUCKET}/{s3_key}", local_path],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        downloaded += 1
        print(f"  ✅ [{i}/{len(metrics_files)}] system_{sys_id}")
    else:
        errors += 1
        print(f"  ❌ [{i}/{len(metrics_files)}] system_{sys_id}: {result.stderr.strip()}")

print(f"\n  Downloaded : {downloaded}")
print(f"  Skipped    : {skipped}")
print(f"  Errors     : {errors}")

# ─────────────────────────────────────────────
# STEP 4: Read all metrics and build summary
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("STEP 3: Building sensor summary table...")
print(f"{'='*60}")

all_metrics = []

for f in sorted(os.listdir(METRICS_DIR)):
    if not f.endswith(".parquet"):
        continue
    try:
        m = pd.read_parquet(os.path.join(METRICS_DIR, f))
        sys_id = int(m['system_id'].iloc[0])
        sensors = m['sensor_name'].tolist()
        units   = m['units'].tolist()

        # Print per system
        print(f"\n  System {sys_id} ({len(sensors)} sensors):")
        for name, unit in zip(sensors, units):
            print(f"    • {name} ({unit})")

        # Store for summary CSV
        for name, unit in zip(sensors, units):
            all_metrics.append({
                'system_id':   sys_id,
                'sensor_name': name,
                'unit':        unit,
            })

    except Exception as e:
        print(f"  ⚠️  Could not read {f}: {e}")

# ─────────────────────────────────────────────
# STEP 5: Save summary CSV
# ─────────────────────────────────────────────
if all_metrics:
    df_summary = pd.DataFrame(all_metrics)

    # Pivot: systems as rows, sensors as columns (✓ if present)
    df_pivot = df_summary.groupby(['system_id', 'sensor_name']).size().unstack(fill_value=0)
    df_pivot = df_pivot.applymap(lambda x: '✓' if x > 0 else '')

    summary_path = os.path.join(BASE_DIR, "metrics_summary.csv")
    df_pivot.to_csv(summary_path)

    print(f"\n{'='*60}")
    print(f"✅ Sensor coverage matrix saved: metrics_summary.csv")
    print(f"\nSensors present across ALL systems:")
    all_cols = df_summary['sensor_name'].value_counts()
    total_systems = df_summary['system_id'].nunique()
    for sensor, count in all_cols.items():
        bar = '█' * count
        print(f"  {sensor:<35} {bar} ({count}/{total_systems} systems)")

print(f"\n{'='*60}")
print("ALL DONE!")
print(f"📁 Metrics folder : {METRICS_DIR}")
print(f"📊 Summary CSV    : {os.path.join(BASE_DIR, 'metrics_summary.csv')}")
print(f"{'='*60}")
