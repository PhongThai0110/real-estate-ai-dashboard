import pandas as pd
import numpy as np

# Danh sách 5 file dữ liệu của bạn
files = {
    "Nhà phố Hồ Chí Minh": "data/data_nha_hcm_final.csv",
    "Nhà phố Hà Nội": "data/data_nha_hn_final.csv",
    "Căn hộ Chung cư": "data/data_apartment_final.csv",
    "Đất nền": "data/data_land_all_final.csv",
    "Biệt thự / Villa": "data/data_villa_vip_final.csv"
}

# Các cột quan trọng cần tính Median
cols_to_calc = [
    'front_width', 'access_road', 'floors', 
    'bedrooms', 'bathrooms', 'land_depth', 'business_potential'
]

print("--- COPY ĐOẠN NÀY VÀO FILE PREPROCESSOR.PY ---")
print("MEDIAN_DEFAULTS = {")

for model_name, file_path in files.items():
    try:
        df = pd.read_csv(file_path)
        stats = {}
        
        # Tính Median cho từng cột (Bỏ qua số 0 vì 0 thường là missing)
        for col in cols_to_calc:
            if col in df.columns:
                # Thay 0 bằng NaN rồi mới tính Median để chính xác hơn
                median_val = df[col].replace(0, np.nan).median()
                
                if not pd.isna(median_val):
                    stats[col] = round(median_val, 1) # Làm tròn 1 số lẻ
        
        # Tính Lat/Lon trung tâm (để backup nếu user không nhập)
        if 'lat' in df.columns:
             stats['lat'] = round(df['lat'].replace(0, np.nan).median(), 4)
        if 'lon' in df.columns:
             stats['lon'] = round(df['lon'].replace(0, np.nan).median(), 4)

        print(f"    '{model_name}': {stats},")
        
    except FileNotFoundError:
        print(f"    # Không tìm thấy file: {file_path}")

print("}")