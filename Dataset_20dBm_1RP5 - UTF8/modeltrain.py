import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import joblib
import os

# Đường dẫn các file .csv
folder = "."  # ← chỉnh lại đường dẫn folder chứa 5 file CSV
files = {
    "center": "center.csv",
    "front_near": "front_near.csv",
    "front_far": "front_far.csv",
    "back_near": "back_near.csv",
    "back_far": "back_far.csv"
}

# Đọc và gộp dữ liệu
df_all = []
for zone, file in files.items():
    path = os.path.join(folder, file)
    df = pd.read_csv(path)
    df.columns = ['EPC', 'AntID', 'RSSI', 'Timestamp']
    df['zone'] = zone
    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)
df_all = df_all[df_all['EPC'].astype(str).str.startswith('ABCF')]
df_all['Timestamp'] = pd.to_datetime(df_all['Timestamp'])
df_all['label'] = (df_all['zone'] == 'center').astype(int)

start_time = df_all['Timestamp'].min()
df_all['time_bucket'] = ((df_all['Timestamp'] - start_time).dt.total_seconds() * 1000 // 500).astype(int)

# Trích đặc trưng
agg = df_all.groupby(['EPC', 'time_bucket', 'label']).agg({
    'RSSI': ['mean', 'std', 'max', 'count'],
    'AntID': pd.Series.nunique
}).reset_index()

agg.columns = ['EPC', 'time_bucket', 'label', 'RSSI_mean', 'RSSI_std', 'RSSI_max', 'read_count', 'antenna_unique']
agg.fillna(0, inplace=True)

X = agg[['RSSI_mean', 'RSSI_std', 'RSSI_max', 'read_count', 'antenna_unique']]
y = agg['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)

# Huấn luyện XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Lưu mô hình
joblib.dump(model, os.path.join(folder, "xgboost_center_classifier.pkl"))
print("✅ Đã lưu model XGBoost tại:", folder)
