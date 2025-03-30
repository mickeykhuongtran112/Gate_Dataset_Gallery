import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# --- 1. Đọc dữ liệu ---
folder = "E:\\Machine-Learning\\Practicing_Trainning\\Classification\\ZoneCenter_Check\\Dataset_20dBm_1RP5"  # ← Cập nhật đường dẫn chứa các file CSV của bạn
files = {
    "center": "center.csv",
    "front_near": "front_near.csv",
    "front_far": "front_far.csv",
    "back_near": "back_near.csv",
    "back_far": "back_far.csv"
}

label_map = {
    "center": 1,
    "front_near": 2,
    "back_near": 2,
    "front_far": 0,
    "back_far": 0
}

dfs = []
for zone, filename in files.items():
    path = os.path.join(folder, filename)
    df = pd.read_csv(path)
    df.columns = ['EPC', 'AntID', 'RSSI', 'Timestamp']
    df['zone'] = label_map[zone]
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df[df['EPC'].astype(str).str.startswith('ABCF')].copy()
start_time = df['Timestamp'].min()
df['time_bucket'] = ((df['Timestamp'] - start_time).dt.total_seconds() * 1000 // 500).astype(int)

# --- 2. Trích đặc trưng ---
agg = df.groupby(['EPC', 'time_bucket', 'zone']).agg({
    'RSSI': ['mean', 'std', 'max', 'count', lambda x: x.max() - x.min()],
    'AntID': pd.Series.nunique
}).reset_index()

agg.columns = ['EPC', 'time_bucket', 'label', 'RSSI_mean', 'RSSI_std', 'RSSI_max', 'read_count', 'RSSI_range', 'antenna_unique']
agg.fillna(0, inplace=True)

# --- 3. Tiền xử lý & Huấn luyện ---
X = agg[['RSSI_mean', 'RSSI_std', 'RSSI_max', 'read_count', 'RSSI_range', 'antenna_unique']]
y = agg['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train, y_train)

# --- 4. Dự đoán ---
y_pred = model.predict(X_test)

# --- 5. Chuyển sang nhị phân để đánh giá "lỗi nghiêm trọng" ---
def convert_to_binary(true, pred):
    true_bin, pred_bin = [], []
    for t, p in zip(true, pred):
        # true
        true_bin.append(1 if t == 1 else 0)
        # pred
        pred_bin.append(1 if p in [1, 2] else 0)
    return true_bin, pred_bin

y_true_bin, y_pred_bin = convert_to_binary(y_test.tolist(), y_pred.tolist())

# --- 6. Đánh giá ---
print("🎯 Classification Report (center vs off-center):")
print(classification_report(y_true_bin, y_pred_bin, digits=3, target_names=["off-center", "center"]))

cm = confusion_matrix(y_true_bin, y_pred_bin)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["off-center", "center"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (binary evaluation with near allowed)")
plt.tight_layout()
plt.show()
