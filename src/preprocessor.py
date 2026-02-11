import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st
import category_encoders as ce
# ==============================================================================
# 1. CẤU HÌNH (CONFIG & MAPPING)
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODER_PATH = os.path.join(BASE_DIR, "models",'encoders', "apartment_project_encoder.pkl")

# --- DANH SÁCH CHO ONE-HOT ENCODING (Dùng cho HN, Đất nền, Villa) ---
# (Phải khớp với tên cột lúc train model của bạn)
ONE_HOT_OPTS = {
    "direction": ["Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc", "Chưa xác định"],
    "legal": ["Sổ hồng/Sổ đỏ", "Hợp đồng mua bán", "Giấy tờ khác", "Vi bằng/Giấy tay"],
    "interior": ["Đầy đủ", "Cơ bản", "Nhà trống", "Thỏa thuận"]
}

# --- MAPPING ĐIỂM SỐ (Dành riêng cho CHUNG CƯ & HCM - nếu dùng số) ---
SCORE_MAPS = {
    "legal_score": {
        "Sổ hồng/Sổ đỏ": 4, "Hợp đồng mua bán": 3, "Hợp đồng mua bán/Chờ sổ": 3,
        "Giấy tờ khác": 2, "Vi bằng/Giấy tay": 1, "Chưa xác định": 2
    },
    "interior_score": {
        "Đầy đủ": 3, "Cơ bản": 2, "Nhà trống": 1, "Thỏa thuận": 1.5, "Chưa xác định": 1.5
    },
    "hcm_interior": {
        "Đầy đủ": 3,
        "Cơ bản": 2,
        "Thỏa thuận": 1,
        "Nhà trống": 0,
        "Chưa xác định": 1 # Fillna(1) như code mẫu
    },
    "hcm_legal": {
        "Sổ hồng/Sổ đỏ": 4,
        "Hợp đồng mua bán": 3,
        "Giấy tờ khác": 2,
        "Vi bằng/Giấy tay": 1,
        "Chưa xác định": 2 # Fillna(2) như code mẫu
    },
    "hanoi_interior": {
        "Đầy đủ": 3,
        "Cơ bản": 2,
        "Thỏa thuận": 1, # Code bạn: Thỏa thuận = 1
        "Nhà trống": 0,
        "Chưa xác định": 1 # Fill mặc định
    },
    "villa_interior": {
        "Đầy đủ": 3, "Cơ bản": 2, "Thỏa thuận": 1, "Nhà trống": 0, "Chưa xác định": 1
    },
    "apartment_legal": {
        "Sổ hồng/Sổ đỏ": 4, 
        "Hợp đồng mua bán": 3, 
        "Hợp đồng mua bán/Chờ sổ": 3,
        "Giấy tờ khác": 2, 
        "Vi bằng/Giấy tay": 1, 
        "Chưa xác định": 2 # Fallback theo code train (fillna(2))
    },
    "apartment_interior": {
        "Đầy đủ": 3, 
        "Cơ bản": 2, 
        "Nhà trống": 1, 
        "Thỏa thuận": 1.5,
        "Chưa xác định": 1.5 # Fallback theo code train (fillna(1.5))
    }
}

BINARY_MAP = {"Có": 1, "Yes": 1, "True": 1, "Không": 0, "No": 0, "False": 0}

# ==============================================================================
# 2. CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS)
# ==============================================================================
# --- GIÁ TRỊ TRUNG VỊ (MEDIAN) TỪ TẬP TRAIN ---
# Dùng để điền khuyết khi người dùng nhập thiếu
MEDIAN_DEFAULTS = {
    'Nhà phố Hồ Chí Minh': {
        'front_width': 4.9, 'access_road': 2.5, 'floors': 3.0, 
        'bedrooms': 4.0, 'bathrooms': 4.0, 
        'lat': 10.7958, 'lon': 106.6748
    },
    'Nhà phố Hà Nội': {
        'front_width': 4.5, 'access_road': 2.5, 'floors': 5.0, 
        'bedrooms': 4.0, 'bathrooms': 4.0, 
        'lat': 21.0191, 'lon': 105.818
    },
    'Căn hộ Chung cư': {
        'bedrooms': 2.0, 'bathrooms': 2.0, 
        'lat': 10.8951, 'lon': 106.6767
    },
    'Đất nền': {
        'front_width': 6.0, 'access_road': 8.0, 
        'land_depth': 19.6, 'business_potential': 52.5, 
        'lat': 11.0802, 'lon': 106.7485
    },
    'Biệt thự / Villa': {
        'front_width': 10.0, 'access_road': 15.0, 
        'floors': 4.0, 'bedrooms': 5.0, 'bathrooms': 5.0, 
        'lat': 16.069, 'lon': 106.6589
    },
}
def apply_one_hot(df, col_name, options):
    """Biến 1 cột thành nhiều cột One-Hot (gán 0 hoặc 1)"""
    if col_name not in df.columns:
        return df
        
    user_val = df[col_name].iloc[0]
    for opt in options:
        # Tên cột mới: Ví dụ 'direction_Đông' (Kiểm tra lại prefix của bạn lúc train nhé)
        new_col = f"{col_name}_{opt}" 
        df[new_col] = 1.0 if user_val == opt else 0.0
        
    return df.drop(columns=[col_name])
# --- CẤU HÌNH ĐỊA LÝ (CHO ĐẤT NỀN) ---
PROVINCE_CENTERS = {
    'HaNoi': (21.0285, 105.8542), 'HCMC': (10.7721, 106.6983),
    'DaNang': (16.0544, 108.2022), 'HaiPhong': (20.8561, 106.6822),
    'CanTho': (10.0452, 105.7469), 'NhaTrang': (12.2388, 109.1967),
    'VungTau': (10.3460, 107.0843), 'BienHoa': (10.9574, 106.8427),
    'ThuDauMot': (10.9805, 106.6519), 'BuonMaThuot': (12.6667, 108.0383),
    'Vinh': (18.6733, 105.6800), 'Hue': (16.4637, 107.5909),
    'DaLat': (11.9404, 108.4583), 'QuyNhon': (13.7820, 109.2192),
    'LongXuyen': (10.3777, 105.4409), 'ThaiNguyen': (21.5942, 105.8482),
    'BacNinh': (21.1861, 106.0763), 'HaLong': (20.9599, 107.0425),
    'ThanhHoa': (19.8073, 105.7763), 'NamDinh': (20.4304, 106.1756),
    'VietTri': (21.3228, 105.4022), 'Pleiku': (13.9833, 108.0000),
    'PhanThiet': (10.9804, 108.0389), 'CaMau': (9.1769, 105.1524)
}

# Đường dẫn file KMeans (Bạn cần lưu file này lúc train nhé!)
KMEANS_PATH = os.path.join(BASE_DIR, "models", "kmeans_land.pkl")

def haversine_np(lat1, lon1, lat2, lon2):
    """Hàm tính khoảng cách (km) giữa 2 điểm tọa độ"""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c
def clean_binary_cols(df):
    """
    Chuyển đổi Có/Không thành 1/0 (Chỉ áp dụng cho các cột Yes/No thực sự)
    """
    for col in df.columns:
        # CHỈ áp dụng cho các cột bắt đầu bằng 'is_' (như is_corner, is_car_accessible)
        # TUYỆT ĐỐI KHÔNG động vào 'business_potential'
        if col.startswith('is_'):
            # Chỉ map nếu dữ liệu đang là chữ (do App gửi về)
            if df[col].dtype == 'object':
                df[col] = df[col].map(BINARY_MAP).fillna(0)
    return df

# ==============================================================================
# 3. CÁC HÀM XỬ LÝ CHUYÊN BIỆT (WORKERS)
# ==============================================================================

def process_apartment(df):
    """
    LOGIC RIÊNG CHO CHUNG CƯ (OPTIMIZED)
    - Input: Dữ liệu thô từ Sidebar (12+ cột)
    - Output: Dữ liệu tinh gọn cho Model (8 cột chuẩn)
    """
    defaults = MEDIAN_DEFAULTS['Căn hộ Chung cư']

    # ==========================================================
    # 1. MAPPING TÊN CỘT (QUAN TRỌNG)
    # ==========================================================
    # Sidebar trả về 'toilet', nhưng Model/Encoder cần 'bathrooms'
    if 'toilet' in df.columns:
        df['bathrooms'] = df['toilet']

    # ==========================================================
    # 2. XỬ LÝ SỐ LIỆU & ĐIỀN KHUYẾT
    # ==========================================================
    # A. Số thực (Area, Lat, Lon)
    float_cols = ['area', 'lat', 'lon']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(defaults.get(col, 0.0))

    # B. Số nguyên (Bedrooms, Bathrooms) - KHÔNG XỬ LÝ FLOORS
    int_cols = ['bedrooms', 'bathrooms']
    for col in int_cols:
        if col in df.columns:
            # Ép kiểu int để khớp với lúc train
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(defaults.get(col, 0)).astype(int)

    # ==========================================================
    # 3. FEATURE ENGINEERING (SCORING)
    # ==========================================================
    
    # A. Legal Score
    if 'legal' in df.columns:
        df['legal_score'] = df['legal'].map(SCORE_MAPS['apartment_legal']).fillna(2).astype(float)
    else:
        df['legal_score'] = 2.0

    # B. Interior Score
    if 'interior' in df.columns:
        df['interior_score'] = df['interior'].map(SCORE_MAPS['apartment_interior']).fillna(1.5).astype(float)
    else:
        df['interior_score'] = 1.5

    # ==========================================================
    # 4. ENCODE TÊN DỰ ÁN (KHẮC PHỤC LỖI DIMENSION)
    # ==========================================================
    # Đây là danh sách 8 cột VÀNG mà Encoder/Model yêu cầu
    # Thứ tự phải chuẩn xác 100%
    target_cols = [
        'area', 'lat', 'lon', 
        'bedrooms', 'bathrooms', 
        'legal_score', 'interior_score', 
        'project_name'
    ]

    # Mặc định gán giá trị trung bình nếu chưa encode được
    # (Giá trị 22.5 tương đương khoảng 6 tỷ sau khi expm1, tránh bị về 0)
    df['project_name_encoded'] = 22.5 

    if 'project_name' in df.columns:
        if os.path.exists(ENCODER_PATH):
            try:
                with open(ENCODER_PATH, "rb") as f:
                    encoder = pickle.load(f)
                
                # --- BƯỚC QUAN TRỌNG NHẤT: LỌC CỘT ---
                # Tạo một DataFrame tạm chỉ chứa đúng 8 cột cần thiết
                # Để Encoder không bị "sốc" khi thấy các cột lạ (front_width, direction...)
                df_clean = df[target_cols].copy()
                
                # Transform trên dữ liệu sạch
                df_encoded = encoder.transform(df_clean)
                
                # Gán lại kết quả
                if 'project_name' in df_encoded.columns:
                    df['project_name_encoded'] = df_encoded['project_name']
                    print(f"✅ Encode OK: {df['project_name_encoded'].iloc[0]}")
                
            except Exception as e:
                print(f"⚠️ Lỗi Encoder: {e}")
                # Giữ nguyên giá trị mặc định 22.5
        else:
             print("⚠️ Không tìm thấy file Encoder")

    # Gán giá trị đã encode vào cột project_name để trả về
    df['project_name'] = df['project_name_encoded']

    # ==========================================================
    # 5. TRẢ VỀ DỮ LIỆU CUỐI CÙNG
    # ==========================================================
    # Đảm bảo chỉ trả về đúng 8 cột mà Model XGBoost cần
    
    # Kiểm tra lần cuối (Safety check)
    for col in target_cols:
        if col not in df.columns:
            df[col] = 0
            
    return df[target_cols]

def process_hcm(df):
    """
    LOGIC RIÊNG CHO NHÀ HCM (Chuẩn theo X_train.info())
    """
    defaults = MEDIAN_DEFAULTS['Nhà phố Hồ Chí Minh']

    # ==========================================================
    # 1. ĐIỀN KHUYẾT (IMPUTATION)
    # ==========================================================

    # A. Xử lý số thực (Float)
    if 'front_width' in df.columns:
        val = df['front_width'].iloc[0]
        if pd.isna(val) or val <= 0:
             df['front_width'] = defaults['front_width']

    if 'access_road' in df.columns:
        val = df['access_road'].iloc[0]
        if pd.isna(val) or val <= 0:
             df['access_road'] = defaults['access_road']

    # B. Xử lý Toilet/Phòng ngủ/Tầng (Int)
    # Trong X_train của bạn bedrooms/bathrooms là float (có thể do có NaN),
    # nhưng logic thực tế nó là số nguyên. Ta fillna và ép kiểu int cho an toàn.
    int_cols = ['floors', 'bedrooms', 'bathrooms']
    for col in int_cols:
        if col in df.columns:
            val = df[col].iloc[0]
            if pd.isna(val) or val <= 0:
                # Logic thông minh cho bathroom
                if col == 'bathrooms' and 'bedrooms' in df.columns:
                    bed_val = df['bedrooms'].iloc[0]
                    if bed_val > 0:
                        df[col] = bed_val
                    else:
                        df[col] = defaults[col]
                else:
                    df[col] = defaults.get(col, 1) # Mặc định tối thiểu 1
            
            # Ép kiểu số nguyên
            df[col] = df[col].astype(int)

    # ==========================================================
    # 2. FEATURE ENGINEERING
    # ==========================================================
    
    # Tính is_car_accessible
    if 'access_road' in df.columns:
        df['is_car_accessible'] = (df['access_road'] >= 4.0).astype(int)
    else:
        df['is_car_accessible'] = 0

    # Lô góc
    if 'is_corner' in df.columns:
        df['is_corner'] = df['is_corner'].fillna(0).astype(int)
    else:
        df['is_corner'] = 0

    # ==========================================================
    # 3. ENCODING
    # ==========================================================
    
    # A. Interior (Ordinal -> Int)
    if 'interior' in df.columns:
        df['interior_encoded'] = df['interior'].map(SCORE_MAPS['hcm_interior']).fillna(1).astype(int)
        df = df.drop(columns=['interior'])
        
    # B. Legal (Ordinal -> Int)
    if 'legal' in df.columns:
        df['legal_score'] = df['legal'].map(SCORE_MAPS['hcm_legal']).fillna(2).astype(int)
        df = df.drop(columns=['legal'])
        
    # C. Direction (One-Hot)
    if 'direction' in df.columns:
        # Bước 1: Tạo đủ tất cả các cột
        df = apply_one_hot(df, 'direction', ONE_HOT_OPTS['direction'])
        
        # Bước 2: XÓA CỘT 'Bắc' (Vì trong X_train không có cột direction_Bắc)
        # Model coi 'Bắc' là trường hợp cơ sở (base case) nên đã drop nó.
        if 'direction_Bắc' in df.columns:
            df = df.drop(columns=['direction_Bắc'])

    # ==========================================================
    # 4. SẮP XẾP CỘT (REORDER) - QUAN TRỌNG NHẤT
    # ==========================================================
    # Thứ tự phải khớp y chang ảnh X_train.info() bạn gửi
    final_order = [
        'area', 'lat', 'lon', 'front_width', 'access_road', 
        'floors', 'bedrooms', 'bathrooms', 
        'is_car_accessible', 'is_corner', 
        'interior_encoded', 'legal_score', 
        'direction_Chưa xác định', 
        'direction_Nam', 
        'direction_Tây', 
        'direction_Tây Bắc', 
        'direction_Tây Nam', 
        'direction_Đông', 
        'direction_Đông Bắc', 
        'direction_Đông Nam'
    ]
    
    # Đảm bảo đủ cột, thiếu thì bù 0 (False)
    for col in final_order:
        if col not in df.columns:
            # Với các cột direction bool, fill 0
            df[col] = 0
            
    # Ép kiểu cho các cột Direction thành bool hoặc int (Khuyến nghị int cho an toàn với XGBoost)
    # Tuy nhiên X_train của bạn là bool, nên ta ép kiểu bool cho giống
    bool_cols = [c for c in final_order if 'direction_' in c]
    for c in bool_cols:
        df[c] = df[c].astype(bool)

    return df[final_order]
def process_hanoi(df):
    """
    LOGIC RIÊNG CHO NHÀ HÀ NỘI (Chuẩn hóa theo X_train.info())
    Features cần có: 
    - area, lat, lon, front_width, access_road (float)
    - floors, bedrooms, bathrooms (int)
    - is_corner, road_class (int)
    - interior_encoded (int)
    - legal_Hợp đồng mua bán, legal_Sổ hồng/Sổ đỏ, legal_Vi bằng/Giấy tay (bool/int)
    """
    defaults = MEDIAN_DEFAULTS['Nhà phố Hà Nội']

    # ==========================================================
    # 1. ĐIỀN KHUYẾT & ÉP KIỂU SỐ (BASIC FILLNA)
    # ==========================================================
    
    # List các cột số thực
    float_cols = ['area', 'lat', 'lon', 'front_width', 'access_road']
    for col in float_cols:
        if col in df.columns:
            val = df[col].iloc[0]
            if pd.isna(val) or val <= 0:
                df[col] = defaults.get(col, 0.0)
            df[col] = df[col].astype(float)

    # List các cột số nguyên (Bắt buộc làm tròn)
    int_cols = ['floors', 'bedrooms', 'bathrooms']
    for col in int_cols:
        if col in df.columns:
            val = df[col].iloc[0]
            if pd.isna(val) or val <= 0:
                df[col] = defaults.get(col, 0)
            # Làm tròn và ép kiểu int (Ví dụ: 2.5 tầng -> 3 tầng)
            df[col] = df[col].round().astype(int)

    # ==========================================================
    # 2. FEATURE ENGINEERING (TẠO CỘT MỚI)
    # ==========================================================

    # A. TÍNH ROAD CLASS (Phân loại đường) - QUAN TRỌNG
    # Logic: Dựa trên độ rộng ngõ (access_road)
    # Bạn kiểm tra lại logic lúc train nhé. Đây là logic phổ biến:
    # 1: Ngõ ba gác (< 2.5m) | 2: Ngõ ô tô (2.5m - 5m) | 3: Mặt phố (> 5m)
    if 'access_road' in df.columns:
        width = df['access_road'].iloc[0]
        if width >= 5.0:
            df['road_class'] = 3
        elif width >= 2.5:
            df['road_class'] = 2
        else:
            df['road_class'] = 1
    else:
        df['road_class'] = 1 # Mặc định ngõ nhỏ

    # B. Xử lý is_corner (Đảm bảo là int)
    if 'is_corner' in df.columns:
        df['is_corner'] = df['is_corner'].fillna(0).astype(int)
    else:
        df['is_corner'] = 0

    # ==========================================================
    # 3. ENCODING (MÃ HÓA)
    # ==========================================================

    # A. Interior (Ordinal Encoding)
    if 'interior' in df.columns:
        df['interior_encoded'] = df['interior'].map(SCORE_MAPS['hanoi_interior']).fillna(1).astype(int)
        df = df.drop(columns=['interior'])

    # B. Legal (One-Hot Encoding - Chỉ lấy đúng 3 cột model cần)
    if 'legal' in df.columns:
        # Bước 1: Bung lụa tạo tất cả các cột One-Hot
        df = apply_one_hot(df, 'legal', ONE_HOT_OPTS['legal'])
        
        # Bước 2: Chỉ giữ lại 3 cột khớp với X_train
        # (Lưu ý: Nếu user chọn 'Giấy tờ khác', cả 3 cột này sẽ bằng 0 -> Đúng logic drop_first)
        required_legal_cols = [
            'legal_Hợp đồng mua bán', 
            'legal_Sổ hồng/Sổ đỏ', 
            'legal_Vi bằng/Giấy tay'
        ]
        
        for col in required_legal_cols:
            if col not in df.columns:
                df[col] = 0 # Tạo cột mới full 0 nếu không có
            
            # Ép kiểu bool (hoặc int 0/1 đều được, int an toàn hơn cho Web)
            df[col] = df[col].astype(int) 

    # C. Xóa các cột thừa (Direction - vì model train không dùng)
    if 'direction' in df.columns:
        df = df.drop(columns=['direction'])
        
    # Xóa các cột legal thừa (Ví dụ: legal_Giấy tờ khác) nếu lỡ tạo ra
    cols_to_remove = [c for c in df.columns if c.startswith('legal_') and c not in required_legal_cols]
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)

    # ==========================================================
    # 4. SẮP XẾP CỘT (REORDER) - ĐỂ KHỚP VỊ TRÍ VỚI MODEL
    # ==========================================================
    # Model sklearn rất khó tính, thứ tự cột phải đúng y chang lúc train
    final_order = [
        'area', 'lat', 'lon', 'front_width', 'access_road', 
        'floors', 'bedrooms', 'bathrooms', 
        'is_corner', 'road_class', 'interior_encoded', 
        'legal_Hợp đồng mua bán', 'legal_Sổ hồng/Sổ đỏ', 'legal_Vi bằng/Giấy tay'
    ]
    
    # Chỉ lấy các cột có trong list trên
    # (Nếu thiếu cột nào thì fill 0 để tránh crash)
    for col in final_order:
        if col not in df.columns:
            df[col] = 0
            
    # Trả về đúng thứ tự
    return df[final_order]
def process_land(df):
    """
    LOGIC PHỨC TẠP CHO ĐẤT NỀN (Chuẩn theo X_train.info())
    """
    defaults = MEDIAN_DEFAULTS['Đất nền']

    # ==========================================================
    # 1. ĐIỀN KHUYẾT (IMPUTATION)
    # ==========================================================
    
    # A. Front Width (Mặt tiền) - Logic Median từ Train
    if 'front_width' in df.columns:
        val = df['front_width'].iloc[0]
        if pd.isna(val) or val <= 0:
            df['front_width'] = defaults['front_width']
            
    # B. Access Road (Đường vào)
    if 'access_road' in df.columns:
        val = df['access_road'].iloc[0]
        if pd.isna(val) or val <= 0:
            df['access_road'] = defaults['access_road']

    # C. Area (Diện tích) - Để tránh log(0)
    if 'area' in df.columns:
         if df['area'].iloc[0] <= 0:
             df['area'] = 50.0 # Giá trị an toàn

    # Đảm bảo lat/lon không bị 0 (dùng default)
    for col in ['lat', 'lon']:
        if col in df.columns:
            if df[col].iloc[0] == 0: df[col] = defaults[col]

    # ==========================================================
    # 2. FEATURE ENGINEERING (HÌNH HỌC & LOGIC)
    # ==========================================================

    # A. Log Area (Thay thế Area gốc)
    df['log_area'] = np.log1p(df['area'])

    # B. Land Depth (Chiều sâu giả định)
    # Công thức: Area / (Width + 0.1)
    df['land_depth'] = df['area'] / (df['front_width'] + 0.1)

    # C. Shape Ratio (Tỷ lệ hình dáng)
    # Công thức: Depth / (Width + 0.1)
    df['shape_ratio'] = df['land_depth'] / (df['front_width'] + 0.1)

    # D. Business Potential (Tiềm năng kinh doanh)
    # Công thức: Width * Access Road
    df['business_potential'] = df['front_width'] * df['access_road']

    # E. Road Type (Binning) - Logic 3 nhóm
    # < 3m: 0 | 3-6m: 1 | > 6m: 2
    width = df['access_road'].iloc[0]
    if width < 3.0:
        df['road_type'] = 0
    elif width <= 6.0:
        df['road_type'] = 1
    else:
        df['road_type'] = 2
    # Ép kiểu int64 cho road_type
    df['road_type'] = df['road_type'].astype(int)

    # ==========================================================
    # 3. FEATURE ENGINEERING (ĐỊA LÝ - DISTANCE & CLUSTER)
    # ==========================================================

    # A. Log Distance (Khoảng cách đến TP gần nhất)
    # Tìm khoảng cách nhỏ nhất trong list PROVINCE_CENTERS
    centers_coords = list(PROVINCE_CENTERS.values()) # List các tuple (lat, lon)
    user_lat = df['lat'].iloc[0]
    user_lon = df['lon'].iloc[0]
    
    # Tính khoảng cách đến TẤT CẢ các tâm, lấy min
    # Lưu ý: Ta tính thủ công thay vì dùng KDTree vì ở đây chỉ có 1 điểm, dùng vòng lặp nhanh hơn load KDTree
    min_dist = float('inf')
    for c_lat, c_lon in centers_coords:
        dist = haversine_np(user_lat, user_lon, c_lat, c_lon)
        if dist < min_dist:
            min_dist = dist
            
    df['log_dist'] = np.log1p(min_dist)

    # B. Geo Cluster (KMeans) - QUAN TRỌNG
    # Bạn phải load model KMeans đã train. Nếu chưa có file, ta gán mặc định cluster 0.
    if os.path.exists(KMEANS_PATH):
        try:
            with open(KMEANS_PATH, "rb") as f:
                kmeans = pickle.load(f)
            # Dự báo cụm
            df['geo_cluster'] = kmeans.predict(df[['lat', 'lon']])[0]
        except:
            df['geo_cluster'] = 0 # Fallback
    else:
        # Nếu user chưa lưu file KMeans -> Gán 0 để app không crash
        # (Lưu ý: Bạn cần train lại và lưu kmeans_land.pkl để chính xác)
        df['geo_cluster'] = 0 
        
    df['geo_cluster'] = df['geo_cluster'].astype(int)

    # ==========================================================
    # 4. ENCODING (ONE-HOT)
    # ==========================================================

    # A. Direction (One-Hot, Drop 'Bắc')
    if 'direction' in df.columns:
        df = apply_one_hot(df, 'direction', ONE_HOT_OPTS['direction'])
        # Drop cột 'direction_Bắc' (Do logic drop_first=True lúc train)
        if 'direction_Bắc' in df.columns:
            df = df.drop(columns=['direction_Bắc'])

    # B. Legal (One-Hot, Drop 'Giấy tờ khác')
    if 'legal' in df.columns:
        df = apply_one_hot(df, 'legal', ONE_HOT_OPTS['legal'])
        # Xóa 'legal_Giấy tờ khác' (Do logic drop_first=True, cột này thường đứng đầu alphabet G < H < S < V)
        # Kiểm tra lại thứ tự alphabet:
        # Giấy tờ khác (G) - Hợp đồng (H) - Sổ hồng (S) - Vi bằng (V)
        # -> Đúng là Giấy tờ khác bị drop.
        if 'legal_Giấy tờ khác' in df.columns:
            df = df.drop(columns=['legal_Giấy tờ khác'])

    # ==========================================================
    # 5. SẮP XẾP CỘT (REORDER & CLEANUP)
    # ==========================================================
    
    # Danh sách cột chuẩn từ X_train.info()
    final_order = [
        'lat', 'lon', 
        'front_width', 'access_road',
        # Nhóm Direction (8 cột - mất Bắc)
        'direction_Chưa xác định', 'direction_Nam', 'direction_Tây',
        'direction_Tây Bắc', 'direction_Tây Nam', 'direction_Đông',
        'direction_Đông Bắc', 'direction_Đông Nam',
        # Nhóm Legal (3 cột - mất Giấy tờ khác)
        'legal_Hợp đồng mua bán', 'legal_Sổ hồng/Sổ đỏ', 'legal_Vi bằng/Giấy tay',
        # Các cột Feature Engineering
        'geo_cluster', 
        'land_depth', 'shape_ratio', 'business_potential',
        'log_area', 'log_dist', 'road_type'
    ]

    # Bù các cột thiếu bằng 0 / False
    for col in final_order:
        if col not in df.columns:
            df[col] = 0
            
    # Ép kiểu bool cho các cột direction/legal (theo X_train)
    bool_cols = [c for c in final_order if 'direction_' in c or 'legal_' in c]
    for c in bool_cols:
        df[c] = df[c].astype(bool)

    return df[final_order]
def process_villa(df):
    """
    LOGIC RIÊNG CHO BIỆT THỰ / VILLA (Chuẩn theo X_train.info())
    - Road Class: mốc 4m và 7m
    - Imputation: Median
    - Encoding: One-Hot (Drop first)
    """
    defaults = MEDIAN_DEFAULTS['Biệt thự / Villa']

    # ==========================================================
    # 1. ĐIỀN KHUYẾT & ÉP KIỂU SỐ (IMPUTATION)
    # ==========================================================
    
    # A. Cột số thực (Float)
    float_cols = ['area', 'lat', 'lon', 'front_width', 'access_road']
    for col in float_cols:
        if col in df.columns:
            val = df[col].iloc[0]
            if pd.isna(val) or val <= 0:
                df[col] = defaults.get(col, 0.0)
            df[col] = df[col].astype(float)

    # B. Cột số nguyên (Int) - Bắt buộc làm tròn
    int_cols = ['floors', 'bedrooms', 'bathrooms']
    for col in int_cols:
        if col in df.columns:
            val = df[col].iloc[0]
            if pd.isna(val) or val <= 0:
                df[col] = defaults.get(col, 0)
            # Làm tròn và ép kiểu int (VD: 2.5 -> 3)
            df[col] = df[col].round().astype(int)

    # ==========================================================
    # 2. FEATURE ENGINEERING (TẠO CỘT MỚI)
    # ==========================================================

    # A. Road Class (Phân loại đường - Logic RIÊNG của Villa)
    # Logic Train: < 4m (0) | < 7m (1) | >= 7m (2)
    if 'access_road' in df.columns:
        width = df['access_road'].iloc[0]
        if pd.isna(width):
             df['road_class'] = 0
        elif width < 4.0:
            df['road_class'] = 0  # Hẻm nhỏ
        elif width < 7.0:
            df['road_class'] = 1  # Xe hơi vào
        else:
            df['road_class'] = 2  # Đường lớn/2 xe tránh
    else:
        df['road_class'] = 0
    
    # Ép kiểu int cho road_class
    df['road_class'] = df['road_class'].astype(int)

    # B. Xử lý is_corner
    if 'is_corner' in df.columns:
        df['is_corner'] = df['is_corner'].fillna(0).astype(int)
    else:
        df['is_corner'] = 0

    # ==========================================================
    # 3. ENCODING (MÃ HÓA)
    # ==========================================================

    # A. Interior (Ordinal Encoding)
    if 'interior' in df.columns:
        df['interior_encoded'] = df['interior'].map(SCORE_MAPS['villa_interior']).fillna(1).astype(int)
        df = df.drop(columns=['interior'])

    # B. Direction (One-Hot Encoding)
    if 'direction' in df.columns:
        df = apply_one_hot(df, 'direction', ONE_HOT_OPTS['direction'])
        # Xóa cột 'direction_Bắc' (Do drop_first=True)
        if 'direction_Bắc' in df.columns:
            df = df.drop(columns=['direction_Bắc'])

    # C. Legal (One-Hot Encoding)
    if 'legal' in df.columns:
        df = apply_one_hot(df, 'legal', ONE_HOT_OPTS['legal'])
        
        # Chỉ giữ lại 3 cột Legal có trong X_train
        required_legal_cols = [
            'legal_Hợp đồng mua bán', 
            'legal_Sổ hồng/Sổ đỏ', 
            'legal_Vi bằng/Giấy tay'
        ]
        
        for col in required_legal_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].astype(bool) # X_train là bool
            
        # Xóa các cột thừa (ví dụ legal_Giấy tờ khác)
        cols_to_remove = [c for c in df.columns if c.startswith('legal_') and c not in required_legal_cols]
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)

    # ==========================================================
    # 4. SẮP XẾP CỘT (REORDER) - BƯỚC CUỐI
    # ==========================================================
    # Thứ tự chuẩn theo X_train.info() của Villa
    final_order = [
        'area', 'lat', 'lon', 'front_width', 'access_road', 
        'floors', 'bedrooms', 'bathrooms', 
        'is_corner', 'interior_encoded', 
        # Direction (8 cột - không có Bắc)
        'direction_Chưa xác định', 'direction_Nam', 'direction_Tây', 
        'direction_Tây Bắc', 'direction_Tây Nam', 'direction_Đông', 
        'direction_Đông Bắc', 'direction_Đông Nam', 
        # Legal (3 cột - không có Giấy tờ khác)
        'legal_Hợp đồng mua bán', 'legal_Sổ hồng/Sổ đỏ', 'legal_Vi bằng/Giấy tay', 
        'road_class'
    ]
    
    # Bù cột thiếu bằng 0
    for col in final_order:
        if col not in df.columns:
            # Nếu là cột boolean (direction/legal) thì fill False
            if 'direction_' in col or 'legal_' in col:
                df[col] = False
            else:
                df[col] = 0
            
    # Ép kiểu bool lại lần cuối cho các cột One-Hot (để chắc chắn khớp Dtype)
    bool_cols = [c for c in final_order if 'direction_' in c or 'legal_' in c]
    for c in bool_cols:
        df[c] = df[c].astype(bool)

    return df[final_order]


# ==============================================================================
# 4. HÀM ĐIỀU PHỐI CHÍNH (DISPATCHER)
# ==============================================================================

# Đổi tên hàm thành transform_input để khớp với app.py
def transform_input(user_input_dict, model_type):
    """
    Hàm tổng nhận dữ liệu từ App và gọi Worker tương ứng.
    Tham số:
      - user_input_dict: Dictionary chứa dữ liệu nhập từ Sidebar
      - model_type: Chuỗi tên loại hình (VD: "Nhà phố Hồ Chí Minh")
    """
    df = pd.DataFrame([user_input_dict])
    
    # 1. Xử lý Binary (Có/Không -> 1/0)
    df = clean_binary_cols(df)
    
    # 2. Ép kiểu số
    numeric_cols = [
         'area', 'front_width', 'floors', 'bedrooms', 'bathrooms',
         'land_depth', 'business_potential', 'access_road', 'lat', 'lon'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. GỌI WORKER TƯƠNG ỨNG
    # Code của bạn đã chia case rất chuẩn, giữ nguyên logic này
    if model_type == "Căn hộ Chung cư":
        df = process_apartment(df)
    elif model_type == "Nhà phố Hồ Chí Minh":
        df = process_hcm(df)
    elif model_type == "Nhà phố Hà Nội":
        df = process_hanoi(df)
    elif model_type == "Đất nền":
        df = process_land(df)
    elif "Biệt thự" in model_type or "Villa" in model_type:
        df = process_villa(df)
        
    # Fill 0 cuối cùng (Safety check)
    df = df.fillna(0)
    
    return df