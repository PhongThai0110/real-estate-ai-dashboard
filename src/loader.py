import streamlit as st
import pandas as pd
import joblib
import os
import pickle
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
DATA_DIR = os.path.join(root_dir, 'data')
MODEL_DIR = os.path.join(root_dir, 'models')

# --- HÀM PHỤ TRỢ: GẮN CLUSTER ---
def apply_kmeans_logic(df, model_filename):
    """
    Hàm nhận vào DataFrame và tên file model KMeans.
    Trả về DataFrame đã có thêm cột 'geo_cluster'.
    """
    if df is None or df.empty:
        return df

    # Đường dẫn file model
    kmeans_path = os.path.join(MODEL_DIR, model_filename)

    # Nếu không có model thì thôi, trả về df gốc
    if not os.path.exists(kmeans_path):
        # print(f"⚠️ Không tìm thấy KMeans: {model_filename}")
        return df

    try:
        # Load Model (Thử joblib trước, pickle sau)
        try:
            kmeans = joblib.load(kmeans_path)
        except:
            with open(kmeans_path, 'rb') as f:
                kmeans = pickle.load(f)

        # Chỉ predict trên các dòng có tọa độ hợp lệ
        valid_mask = df['lat'].notna() & df['lon'].notna() & (df['lat'] != 0)
        valid_data = df.loc[valid_mask]

        if not valid_data.empty:
            # Sklearn yêu cầu input là array 2 cột [[lat, lon]]
            coords = valid_data[['lat', 'lon']]
            clusters = kmeans.predict(coords)

            # Gán ngược lại vào DF gốc
            df.loc[valid_mask, 'geo_cluster'] = clusters
            
            # Fill các dòng lỗi bằng -1
            df['geo_cluster'] = df['geo_cluster'].fillna(-1).astype(int)
        
        return df

    except Exception as e:
        print(f"⚠️ Lỗi khi chạy KMeans {model_filename}: {e}")
        return df

# --- 1. HÀM LOAD DỮ LIỆU (CHO DASHBOARD) ---
@st.cache_data
def load_raw_data():
    """
    Load CSV và tự động gắn thêm cột 'geo_cluster'
    """
    data = {}
    
    # Mapping: Key -> (Tên file CSV, Tên file KMeans tương ứng)
    # Lưu ý: Tên file KMeans phải khớp với trong folder models của bạn
    config = {
        'hcm':       {'csv': 'data_nha_hcm_final.csv',     'kmeans': 'kmeans_hcm.pkl'},
        'hanoi':     {'csv': 'data_nha_hn_final.csv',      'kmeans': 'kmeans_hanoi.pkl'},
        'apartment': {'csv': 'data_apartment_final.csv',   'kmeans': 'kmeans_apartment.pkl'},
        'land':      {'csv': 'data_land_all_final.csv',    'kmeans': 'kmeans_land.pkl'},
        'villa':     {'csv': 'data_villa_vip_final.csv',   'kmeans': 'kmeans_villa.pkl'}
    }
    
    for key, cfg in config.items():
        csv_path = os.path.join(DATA_DIR, cfg['csv'])
        
        if os.path.exists(csv_path):
            try:
                # 1. Đọc CSV
                df = pd.read_csv(csv_path)
                
                # 2. Chuẩn hóa tên cột (Để tránh lỗi District/district)
                df.columns = [c.lower() for c in df.columns]

                # 3. Gắn Geo Cluster (Chạy KMeans)
                # Đây chính là bước bạn đang thiếu ở file loader cũ!
                df = apply_kmeans_logic(df, cfg['kmeans'])
                
                data[key] = df
            except Exception as e:
                st.error(f"Lỗi đọc {cfg['csv']}: {e}")
                data[key] = pd.DataFrame()
        else:
            data[key] = pd.DataFrame() 
            
    return data

# --- 2. HÀM LOAD MODEL (CHO DỰ BÁO) ---
@st.cache_resource
def load_models(city_mode, property_type):
    """
    Load Model XGBoost VÀ KMeans cho phần dự báo giá
    """
    resources = {}
    
    # Mapping Logic (Giữ nguyên như cũ)
    if property_type == "Nhà phố":
        if city_mode == "Hồ Chí Minh":
            model_file = "best_xgboost_HouseHCM.pkl"
            kmeans_file = "kmeans_hcm.pkl"
        else:
            model_file = "best_xgboost_HanoiHouse.pkl"
            kmeans_file = "kmeans_hanoi.pkl"
    elif property_type == "Căn hộ Chung cư" or property_type == "Chung cư":
        model_file = "best_xgboost_Apartment.pkl"
        kmeans_file = "kmeans_apartment.pkl"
    elif property_type == "Đất nền":
        model_file = "best_xgboost_landall.pkl"
        kmeans_file = "kmeans_land.pkl"
    else: 
        model_file = "best_xgboost_villavip.pkl"
        kmeans_file = "kmeans_villa.pkl"

    # Load XGBoost
    model_path = os.path.join(MODEL_DIR, model_file)
    if os.path.exists(model_path):
        try:
            resources['model'] = joblib.load(model_path)
        except:
            with open(model_path, 'rb') as f:
                resources['model'] = pickle.load(f)
    else:
        return None

    # Load KMeans
    kmeans_path = os.path.join(MODEL_DIR, kmeans_file)
    if os.path.exists(kmeans_path):
        try:
            resources['kmeans'] = joblib.load(kmeans_path)
        except:
            with open(kmeans_path, 'rb') as f:
                resources['kmeans'] = pickle.load(f)
    else:
        resources['kmeans'] = None

    return resources
# --- THÊM VÀO CUỐI FILE src/loader.py ---

@st.cache_data
def get_project_list():
    """
    Hàm lấy danh sách tên dự án duy nhất từ file data_apartment_final.csv
    Để hiển thị lên Sidebar cho người dùng chọn.
    """
    try:
        file_path = os.path.join(DATA_DIR, 'data_apartment_final.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Ưu tiên lấy cột tên dự án dạng chữ (Raw)
            if 'project_name_raw' in df.columns:
                projects = df['project_name_raw'].unique().tolist()
            elif 'project_name' in df.columns:
                projects = df['project_name'].unique().tolist()
            else:
                return []
            
            # Sắp xếp A-Z và loại bỏ giá trị nan
            projects = [str(p) for p in projects if str(p) != 'nan']
            projects.sort()
            return projects
        else:
            return []
    except Exception as e:
        print(f"Lỗi lấy danh sách dự án: {e}")
        return []