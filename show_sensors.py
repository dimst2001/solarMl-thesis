"""
PVDAQ — Show All Systems Sensors
==================================
Reads all metrics parquet files and prints:
- System ID
- System name (if available)
- All sensors with units

HOW TO RUN:
    python3 show_sensors.py
"""

import os
import pandas as pd

HOME        = os.path.expanduser("~")
METRICS_DIR = os.path.join(HOME, "Desktop", "solar-data-all", "metrics")

print("=" * 60)
print("PVDAQ — System Sensors Overview")
print(f"Reading from: {METRICS_DIR}")
print("=" * 60)

files = sorted([f for f in os.listdir(METRICS_DIR) if f.endswith(".parquet")])
print(f"Found {len(files)} metric files\n")

all_rows = []

for f in files:
    try:
        m = pd.read_parquet(os.path.join(METRICS_DIR, f))

        # Get system ID
        sys_id = int(m['system_id'].iloc[0])

        # Get system name if column exists
        name_col = next((c for c in m.columns if 'name' in c.lower()
                         and c != 'sensor_name' and c != 'standard_name'), None)
        sys_name = m[name_col].iloc[0] if name_col else "N/A"

        # Print system header
        print(f"{'='*60}")
        print(f"System ID : {sys_id}")
        print(f"Name      : {sys_name}")
        print(f"Sensors   : {len(m)}")
        print(f"{'-'*60}")

        for _, row in m.iterrows():
            sensor = row.get('sensor_name', 'unknown')
            unit   = row.get('units', '')
            common = row.get('common_name', '')
            print(f"  • {sensor:<35} [{unit}]   {common}")

            all_rows.append({
                'system_id':   sys_id,
                'system_name': sys_name,
                'sensor_name': sensor,
                'unit':        unit,
                'common_name': common,
            })

    except Exception as e:
        print(f"⚠️  Could not read {f}: {e}")

# ─────────────────────────────────────────────
# Save summary CSV
# ─────────────────────────────────────────────
if all_rows:
    df = pd.DataFrame(all_rows)
    out = os.path.join(HOME, "Desktop", "solar-data-all", "metrics_summary.csv")
    df.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print(f"✅ Summary saved: metrics_summary.csv")
    print(f"\nSensor coverage across all {df['system_id'].nunique()} systems:")
    print(f"{'-'*60}")
    counts = df['sensor_name'].value_counts()
    total  = df['system_id'].nunique()
    for sensor, count in counts.items():
        bar = '█' * count
        print(f"  {sensor:<35} {bar} ({count}/{total})")
