import pandas as pd
from sklearn.ensemble import IsolationForest

def load_and_clean_data(file_path="data/financial_anomaly_data.csv"):
    """Đọc dữ liệu sổ cái tài chính và chuẩn hóa thời gian"""
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    return df

def detect_anomalies(df, contamination_rate=0.05):
    """Huấn luyện mô hình Isolation Forest để phát hiện các giao dịch bất thường"""
    X = df[['Amount']].values
    
    # Khởi tạo mô hình học máy phát hiện phần tử ngoại lai
    model = IsolationForest(contamination=contamination_rate, random_state=42)
    
    # Dự đoán (-1 là bất thường, 1 là bình thường)
    df['Anomaly_Score'] = model.fit_predict(X)
    df['Status'] = df['Anomaly_Score'].apply(lambda x: 'Bất thường (Anomaly)' if x == -1 else 'Bình thường')
    
    return df
