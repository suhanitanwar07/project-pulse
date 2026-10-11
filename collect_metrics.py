import csv
import time
from datetime import datetime, timezone

import psutil

OUTPUT_FILE = "real_metrics.csv"
SAMPLES = 30
INTERVAL_SECONDS = 2

columns = [
    "timestamp",
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "network_sent_bytes",
    "network_received_bytes",
]

print("PROJECT PULSE — REAL MAC METRICS")
print("Collecting measurements for approximately 60 seconds...")

with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for sample in range(SAMPLES):
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        network = psutil.net_io_counters()

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": cpu,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "network_sent_bytes": network.bytes_sent,
            "network_received_bytes": network.bytes_recv,
        }

        writer.writerow(row)
        file.flush()

        print(
            f"{sample + 1:02}/{SAMPLES} | "
            f"CPU: {cpu:5.1f}% | "
            f"Memory: {memory.percent:5.1f}% | "
            f"Disk: {disk.percent:5.1f}%"
        )

        if sample < SAMPLES - 1:
            time.sleep(INTERVAL_SECONDS)

print(f"\nDone! Real measurements saved to {OUTPUT_FILE}")
