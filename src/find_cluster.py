import pandas as pd
from sklearn.cluster import KMeans
import pickle
import os

# Cấu hình đường dẫn (Sửa lại cho đúng máy bạn)
DATA_DIR = 'data/'  # Thư mục chứa file csv
MODEL_DIR = 'models/' # Thư mục chứa file pkl

# Đảm bảo thư mục models tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_kmeans(csv_name, pkl_name, n_clusters=30):
    """
    Hàm đọc CSV, train KMeans và lưu file .pkl
    n_clusters=30: Chia thành phố thành 30 khu vực nhỏ (tương đương phường/xã)
    """
    csv_path = os.path.join(DATA_DIR, csv_name)
    pkl_path = os.path.join(MODEL_DIR, pkl_name)
    
    print(f"🔄 Đang xử lý: {csv_name}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ Lỗi: Không tìm thấy file {csv_name}")
        return

    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    # 2. Lọc sạch tọa độ rác (Quan trọng!)
    # Chỉ lấy tọa độ hợp lệ ở Việt Nam
    df = df[(df['lat'] > 8) & (df['lat'] < 24) & (df['lon'] > 102) & (df['lon'] < 110)]
    
    if len(df) == 0:
        print("⚠️ File không có dữ liệu tọa độ hợp lệ!")
        return

    # 3. Train KMeans
    # Chỉ cần 2 cột Lat/Lon
    X = df[['lat', 'lon']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # 4. Lưu file
    with open(pkl_path, 'wb') as f:
        pickle.dump(kmeans, f)
        
    print(f"✅ Đã lưu thành công: {pkl_name} (Train trên {len(df)} dòng)")

# --- CHẠY ---
if __name__ == "__main__":
    # 1. Tạo KMeans cho HCM
    train_and_save_kmeans('data_nha_hcm_final.csv', 'kmeans_hcm.pkl', n_clusters=24) 
    # (HCM có khoảng 24 quận huyện -> chọn 24 cụm là đẹp)

    # 2. Tạo KMeans cho Hà Nội
    train_and_save_kmeans('data_nha_hn_final.csv', 'kmeans_hanoi.pkl', n_clusters=30)
    # (Hà Nội rộng hơn, chọn 30 cụm)
    
    # 3. Tạo KMeans cho Chung cư (Apartment)
    # Chung cư thường rải rác khắp thành phố, cần số cụm lớn
    train_and_save_kmeans('data_apartment_final.csv', 'kmeans_apartment.pkl', n_clusters=50)

    # 4. Tạo KMeans cho Biệt thự (Villa VIP)
    # Biệt thự thường tập trung ở vài khu vực "nhà giàu" (Thảo Điền, Q7, Tây Hồ...)
    # Nên chọn số cụm ít hơn để gom nhóm chính xác hơn
    train_and_save_kmeans('data_villa_vip_final.csv', 'kmeans_villa.pkl', n_clusters=20)

    print("\n🎉 HOÀN TẤT! Hãy copy các file .pkl vào thư mục models/ của App.")