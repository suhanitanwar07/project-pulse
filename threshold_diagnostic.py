
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

data_file = Path(
    "nab-data/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
)
label_file = Path("nab-data/labels/combined_windows.json")

df = pd.read_csv(data_file, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

with label_file.open() as f:
    labels = json.load(f)

windows = labels[
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
]

df["actual"] = 0
for start, end in windows:
    mask = df["timestamp"].between(
        pd.to_datetime(start), pd.to_datetime(end)
    )
    df.loc[mask, "actual"] = 1

previous = df["value"].shift(1)
df["change"] = df["value"] - previous
df["rolling_mean"] = previous.rolling(12, min_periods=12).mean()
df["rolling_std"] = previous.rolling(12, min_periods=12).std()

features = ["value", "change", "rolling_mean", "rolling_std"]
df[features] = df[features].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features).reset_index(drop=True)

split = int(len(df) * 0.70)
train = df.iloc[:split]
test = df.iloc[split:]

if train["actual"].sum() != 0 or test["actual"].sum() == 0:
    raise ValueError("Unexpected split: check the anomaly labels.")

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
)
model.fit(train[features])

# Larger scores mean more anomalous.
train_scores = -model.decision_function(train[features])
test_scores = -model.decision_function(test[features])
y_true = test["actual"].to_numpy()

print("THRESHOLD SENSITIVITY — DIAGNOSTIC ONLY")
print("Thresholds come from training scores, not test labels.")
print(f"Test rows: {len(test)}")
print(f"Labelled anomalies: {y_true.sum()}")
print()
print("Train-score percentile | Precision | Recall | F1")

for percentile in [99, 97, 95, 90, 80]:
    threshold = np.percentile(train_scores, percentile)
    predictions = (test_scores > threshold).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(y_true, predictions, zero_division=0)

    print(
        f"{percentile:>21}% | "
        f"{precision:9.3f} | {recall:6.3f} | {f1:5.3f}"
    )

