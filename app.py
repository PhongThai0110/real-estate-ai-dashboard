import streamlit as st
import pandas as pd
import sys
import os
import numpy as np
# --- KẾT NỐI MODULE ---
from views import sidebar
from views import dashboard  # File View (Vẽ biểu đồ)
from src import loader       # File Model (Load dữ liệu/AI)
from src import preprocessor # File Xử lý dữ liệu đầu vào

# ==============================================================================
# 1. CẤU HÌNH TRANG
# ==============================================================================
def local_css(file_name):
    try:
        with open(file_name,encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Nếu chưa có file css thì thôi, không lỗi
st.set_page_config(
    page_title="Real Estate AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
local_css("assets/style.css")
# ==============================================================================
# 2. HÀM LOGIC DỰ BÁO (AI PREDICTION FLOW)
# ==============================================================================
def execute_prediction_flow(user_inputs, city_mode, property_type):
    """
    Hàm điều phối luồng dự báo giá:
    1. Xác định key xử lý
    2. Load Model XGBoost
    3. Gọi Preprocessor xử lý dữ liệu
    4. Khớp cột & Debug lỗi thiếu cột
    5. Trả về kết quả dự báo (đã chuyển từ Log -> Giá thực)
    """
    
    # --- BƯỚC 1: TẠO KEY CHO PREPROCESSOR ---
    # Key này phải khớp chính xác với các if/elif trong preprocessor.transform_input
    if property_type == "Nhà phố":
        process_key = f"Nhà phố {city_mode}"  # VD: "Nhà phố Hồ Chí Minh"
    elif property_type == "Căn hộ Chung cư": # Lưu ý: Sidebar trả về "Căn hộ Chung cư" chứ không phải "Chung cư"
        process_key = "Căn hộ Chung cư"
    elif property_type == "Đất nền":
        process_key = "Đất nền"
    else:
        process_key = "Biệt thự / Villa"

    # --- BƯỚC 2: LOAD MODEL DỰ BÁO ---
    system_resources = loader.load_models(city_mode, property_type)
    
    if not system_resources or 'model' not in system_resources:
        st.error("❌ Không tìm thấy Model. Hãy kiểm tra folder models/.")
        return None

    model = system_resources['model']

    # --- BƯỚC 3: XỬ LÝ INPUT (PREPROCESSING) ---
    try:
        processed_df = preprocessor.transform_input(user_inputs, process_key)
    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
        return None
    
    # --- BƯỚC 4: KHỚP CỘT & DEBUG (QUAN TRỌNG) ---
    try:
        if hasattr(model, 'feature_names_in_'):
            required_cols = list(model.feature_names_in_)
            current_cols = list(processed_df.columns)
            
            # 1. Tìm & Fill cột thiếu bằng 0
            missing_cols = []
            for col in required_cols:
                if col not in current_cols:
                    processed_df[col] = 0
                    missing_cols.append(col)
            
            # 2. In Debug ra Terminal (Để bạn kiểm tra xem có thiếu cột quan trọng không)
            if missing_cols:
                print("\n" + "="*40)
                print(f"⚠️ CẢNH BÁO: Model {process_key} thiếu các cột sau (đã fill 0):")
                print(missing_cols)
                
                # Kiểm tra giá trị các cột quan trọng
                if property_type == "Căn hộ Chung cư":
                    pj_val = processed_df.get('project_name', pd.Series([0])).iloc[0]
                    floor_val = processed_df.get('floors', pd.Series([0])).iloc[0]
                    print(f"🧐 Project Name Value: {pj_val}")
                    print(f"🧐 Floors Value: {floor_val}")
                print("="*40 + "\n")

            # 3. Sắp xếp đúng thứ tự cột của Model
            processed_df = processed_df[required_cols]
    except Exception as e:
        print(f"Lỗi khớp cột: {e}")
        pass

    # --- BƯỚC 5: DỰ BÁO & CHUYỂN ĐỔI ---
    try:
        # Dự báo (Kết quả là Logarit)
        pred_log = model.predict(processed_df)[0]
        
        # Chuyển về giá thực (Anti-Log)
        pred_real = np.expm1(pred_log) 
    
        return max(0, pred_real)
    except Exception as e:
        st.error(f"Lỗi khi model dự báo: {e}")
        return None

def format_currency(amount):
    if amount >= 1: return f"{amount:,.2f} Tỷ"
    return f"{amount*1000:,.0f} Triệu"

# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH (MAIN)
# ==============================================================================
def main():
    # 1. Hiển thị Sidebar & Lấy Input
    nav_mode, user_inputs, dashboard_category, property_type, submit_btn = sidebar.show_sidebar()

    # ==========================================================================
    # A. CHẾ ĐỘ DASHBOARD PHÂN TÍCH
    # ==========================================================================
    if nav_mode == "📊 Dashboard Phân tích":
        st.title(f"📊 Phân tích: {dashboard_category}")
        
        # 1. LOAD DỮ LIỆU TỪ LOADER (Đã có KMeans và chuẩn hóa cột)
        all_data = loader.load_raw_data()
        
        # 2. MAPPING: Chọn đúng DataFrame dựa trên lựa chọn ở Sidebar
        map_key = {
            "Nhà phố Hồ Chí Minh": "hcm",
            "Nhà phố Hà Nội": "hanoi",
            "Căn hộ Chung cư": "apartment",
            "Đất nền": "land",
            "Biệt thự / Villa": "villa"
        }
        
        selected_key = map_key.get(dashboard_category)
        df_selected = all_data.get(selected_key)

        # 3. HIỂN THỊ GIAO DIỆN (DELEGATE TO VIEW)
        # Thay vì viết code vẽ loằng ngoằng ở đây, ta gọi hàm chuyên dụng bên dashboard.py
        if df_selected is not None and not df_selected.empty:
            dashboard.show_dashboard_ui(df_selected, dashboard_category)
        else:
            st.warning(f"⚠️ Không tìm thấy dữ liệu cho **{dashboard_category}**.")
            st.info("Gợi ý: Kiểm tra file CSV trong thư mục 'data/' hoặc logic trong 'src/loader.py'")

    # ==========================================================================
    # B. CHẾ ĐỘ DỰ BÁO GIÁ (AI PREDICTION)
    # ==========================================================================
    else:
        st.title("🤖 AI Định giá Bất động sản")
        
        # Hiển thị ảnh Banner nếu chưa bấm nút
        if not submit_btn:
            st.info("👈 Vui lòng nhập thông tin BĐS bên thanh Sidebar để bắt đầu định giá.")
            if os.path.exists("assets/banner_intro.png"):
                st.image("assets/banner_intro.png", width="stretch") # Banner dùng use_container_width ok
        else:
            # Kiểm tra input cơ bản
            if user_inputs['area'] <= 0:
                st.error("⚠️ Diện tích phải lớn hơn 0 m².")
            else:
                # Gọi hàm dự báo
                with st.spinner("AI đang phân tích và định giá..."):
                    price = execute_prediction_flow(user_inputs, dashboard_category, property_type)
                
                # Hiển thị kết quả
                if price:
                    st.success("✅ Dự báo thành công!")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Giá tham khảo", format_currency(price))
                    with c2:
                        don_gia = (price * 1000) / user_inputs['area']
                        st.metric("Đơn giá ước tính", f"{don_gia:,.1f} Tr/m²")
                    
                    st.caption("*Kết quả chỉ mang tính chất tham khảo dựa trên dữ liệu quá khứ.*")

if __name__ == "__main__":
    main()