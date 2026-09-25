"""
PVDAQ 20-System Downloader — List then Download
================================================
1. Uses 'aws s3 ls' to find exactly which files exist per system
2. Downloads only confirmed files to one flat folder
3. No guessing filenames — only downloads what's actually there

HOW TO RUN:  python3 download_list_then_cp.py
RESUMABLE:   Safe to stop and restart anytime
"""

import os
import subprocess

HOME = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME, "Desktop", "solar-data-all")
PVDATA_DIR = os.path.join(BASE_DIR, "pvdata")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
BUCKET = "oedi-data-lake"

os.makedirs(PVDATA_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

SYSTEMS = [
    '4', '10', '33', '34', '35', '36', '50', '51',
    '1199', '1200', '1201', '1202', '1203', '1207',
    '1208', '1276', '1277', '1283', '1284', '1289'
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def ls_recursive(prefix):
    """
    List all files under an S3 prefix using aws s3 ls --recursive.
    Returns list of full S3 keys.
    """
    result = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", "--recursive",
         f"s3://{BUCKET}/{prefix}"],
        capture_output=True, text=True
    )
    keys = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 4:
            keys.append(parts[3])  # the S3 key is the 4th column
    return keys

def cp(s3_key, local_path):
    """Download one file, skip if already exists."""
    if os.path.isfile(local_path):
        return "skipped"
    result = subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request",
         f"s3://{BUCKET}/{s3_key}", local_path],
        capture_output=True, text=True
    )
    return "ok" if result.returncode == 0 else f"error: {result.stderr.strip()}"

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

print("=" * 60)
print("PVDAQ Downloader — List then Download")
print(f"Saving all files to: {PVDATA_DIR}")
print("=" * 60)

grand_ok = 0
grand_skipped = 0
grand_errors = 0

for idx, system_id in enumerate(SYSTEMS, 1):

    print(f"\n{'='*60}")
    print(f"[{idx}/{len(SYSTEMS)}] System {system_id}")
    print(f"{'='*60}")

    # ── Step 1: Download metrics ──────────────────────────
    metrics_key   = f"pvdaq/parquet/metrics/metrics__system_{system_id}__part000.parquet"
    metrics_local = os.path.join(METRICS_DIR, f"metrics__system_{system_id}.parquet")
    r = cp(metrics_key, metrics_local)
    print(f"  Metrics: {r}")

    # ── Step 2: List all pvdata files for this system ─────
    prefix = f"pvdaq/parquet/pvdata/system_id={system_id}/"
    print(f"  🔍 Listing files...")
    keys = ls_recursive(prefix)

    if not keys:
        print(f"  ⚠️  No files found for system {system_id}")
        continue

    print(f"  📦 Found {len(keys)} files — starting download...")

    # ── Step 3: Download each confirmed file ──────────────
    sys_ok = 0
    sys_skipped = 0
    sys_errors = 0

    for i, key in enumerate(keys, 1):
        # Flat filename — just the basename
        filename = os.path.basename(key)
        local_path = os.path.join(PVDATA_DIR, filename)

        r = cp(key, local_path)

        if r == "ok":
            sys_ok += 1
            grand_ok += 1
        elif r == "skipped":
            sys_skipped += 1
            grand_skipped += 1
        else:
            sys_errors += 1
            grand_errors += 1
            print(f"  ❌ {filename}: {r}")

        # Progress every 100 files
        if i % 100 == 0 or i == len(keys):
            print(f"  ↳ {i}/{len(keys)} — "
                  f"✅ {sys_ok} downloaded, "
                  f"⏭️  {sys_skipped} skipped, "
                  f"❌ {sys_errors} errors")

    print(f"\n  ✅ System {system_id} done: "
          f"{sys_ok} new, {sys_skipped} already existed")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print(f"\n{'='*60}")
print("ALL DONE!")
print(f"  ✅ Downloaded : {grand_ok} files")
print(f"  ⏭️  Skipped   : {grand_skipped} files")
print(f"  ❌ Errors     : {grand_errors} files")
print(f"📁 All files in: {PVDATA_DIR}")
print(f"▶️  Next step  : python3 process_dataset.py")
print(f"{'='*60}")
