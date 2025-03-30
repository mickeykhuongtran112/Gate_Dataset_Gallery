import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# --- Đọc dữ liệu ---
folder = "E:\\Machine-Learning\\Practicing_Trainning\\Classification\\ZoneCenter_Check\\Dataset_20dBm_1RP5"  # Đường dẫn folder
files = {
    "center": "center.csv",
    "front_far": "front_far.csv",
    "front_near": "front_near.csv",
    "back_far": "back_far.csv",
    "back_near": "back_near.csv"
}

dfs = []
for zone, filename in files.items():
    path = os.path.join(folder, filename)
    df = pd.read_csv(path)
    df.columns = ['EPC', 'AntID', 'RSSI', 'Timestamp']
    df['zone'] = zone
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df[df['EPC'].astype(str).str.startswith('ABCF')].copy()
df['label'] = (df['zone'] == 'center').astype(int)

start_time = df['Timestamp'].min()
df['time_bucket'] = ((df['Timestamp'] - start_time).dt.total_seconds() * 1000 // 500).astype(int)

# --- Trích đặc trưng ---
agg = df.groupby(['EPC', 'time_bucket', 'label']).agg({
    'RSSI': ['mean', 'std', 'max', 'count'],
    'AntID': pd.Series.nunique
}).reset_index()
agg.columns = ['EPC', 'time_bucket', 'label', 'RSSI_mean', 'RSSI_std', 'RSSI_max', 'read_count', 'antenna_unique']
agg.fillna(0, inplace=True)

# --- Huấn luyện ---
X = agg[['RSSI_mean', 'RSSI_std', 'RSSI_max', 'read_count', 'antenna_unique']]
y = agg['label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# --- Tối ưu ngưỡng ---
y_probs = model.predict_proba(X_test)[:, 1]
prec, rec, thresh = precision_recall_curve(y_test, y_probs)
f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
best_idx = f1s.argmax()
best_thresh = thresh[best_idx]
y_pred = (y_probs >= best_thresh).astype(int)

# --- Hiển thị kết quả ---
print("✅ Threshold tối ưu:", round(best_thresh, 3))
print("🎯 Classification Report (Lớp 'center'):")
print(classification_report(y_test, y_pred, digits=3))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Off-center", "Center"])
disp.plot(cmap='Blues')
plt.title(f"XGBoost Confusion Matrix (Thresh={round(best_thresh,2)})")
plt.tight_layout()
plt.show()
