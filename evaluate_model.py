
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA_FILE = Path(
    "nab-data/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
)
LABEL_FILE = Path("nab-data/labels/combined_windows.json")
OUTPUT_FILE = Path("nab_test_predictions.csv")

# Load the real AWS CloudWatch time series.
df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Load NAB's published anomaly windows.
with LABEL_FILE.open() as file:
    all_windows = json.load(file)

windows = all_windows[
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
]

df["actual_anomaly"] = 0

for start, end in windows:
    start_time = pd.to_datetime(start)
    end_time = pd.to_datetime(end)

    mask = df["timestamp"].between(start_time, end_time)
    df.loc[mask, "actual_anomaly"] = 1

# Build features using previous measurements only.
# This avoids using future observations to calculate features.
previous = df["value"].shift(1)

df["previous_value"] = previous
df["change"] = df["value"] - previous
df["rolling_mean"] = previous.rolling(12, min_periods=12).mean()
df["rolling_std"] = previous.rolling(12, min_periods=12).std()

feature_columns = [
    "value",
    "change",
    "rolling_mean",
    "rolling_std",
]

df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

df = df.dropna(subset=feature_columns).reset_index(drop=True)

# Chronological split: earlier observations for training,
# later observations for testing.
split_index = int(len(df) * 0.70)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

X_train = train[feature_columns]
X_test = test[feature_columns]
y_test = test["actual_anomaly"].astype(int)

print("PROJECT PULSE — NAB REAL-DATA EVALUATION")
print("=" * 48)
print(f"Dataset rows: {len(df)}")
print(f"Training rows: {len(train)}")
print(f"Testing rows: {len(test)}")
print(f"Anomalies in training: {train['actual_anomaly'].sum()}")
print(f"Anomalies in testing: {y_test.sum()}")

if train["actual_anomaly"].sum() > 0:
    raise ValueError(
        "Training contains labelled anomalies. "
        "Choose a training period before the first anomaly window."
    )

if y_test.sum() == 0:
    raise ValueError(
        "The test period contains no labelled anomalies."
    )

# Fit the model only on the earlier training period.
model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
)
model.fit(X_train)

# Isolation Forest returns -1 for anomalies and 1 for normal points.
test["predicted_anomaly"] = (
    model.predict(X_test) == -1
).astype(int)

y_pred = test["predicted_anomaly"]

print("\nCONFUSION MATRIX")
print("Rows = actual [normal, anomaly]")
print("Columns = predicted [normal, anomaly]")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )
)

print("\nANOMALY METRICS")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
print(f"F1 score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")

# Save predictions so we can inspect detections and missed anomalies.
test.to_csv(OUTPUT_FILE, index=False)
print(f"\nTest predictions saved to {OUTPUT_FILE}")

