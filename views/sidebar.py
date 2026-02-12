import streamlit as st
import os
from src import loader # Import loader để lấy danh sách dự án
from geopy.geocoders import MapBox # <--- THÊM DÒNG NÀY
from time import sleep                # <--- THÊM DÒNG NÀY
def show_sidebar():
    with st.sidebar:
        # 1. LOGO
        if os.path.exists("assets/logo_fpt.png"):
            st.image("assets/logo_fpt.png", width=150)
        
        st.header("🏡 Real Estate AI")
        
        # 2. MENU
        nav_mode = st.radio("Chế độ:", ["🔍 Dự báo giá nhà", "📊 Dashboard Phân tích"])
        st.markdown("---")
        
        # === A. CHẾ ĐỘ DASHBOARD ===
        if nav_mode == "📊 Dashboard Phân tích":
            st.subheader("Tùy chọn hiển thị")
            dashboard_category = st.selectbox(
                "Chọn dữ liệu:", 
                ["Nhà phố Hồ Chí Minh", "Nhà phố Hà Nội", "Căn hộ Chung cư", "Đất nền", "Biệt thự / Villa"]
            )
            st.info("👨‍💻 **Thái Nguyên Phong**\n\nSinh viên AI - Năm 2\nFPT University")
            return nav_mode, {}, dashboard_category, "All", False

        # === B. CHẾ ĐỘ DỰ BÁO ===
        else:
            st.subheader("📝 Nhập thông tin BĐS")
            
            # 1. Chọn Loại hình & Khu vực trước
            property_type = st.selectbox("Loại hình BĐS", ["Nhà phố", "Căn hộ Chung cư", "Đất nền", "Biệt thự / Villa"])
            
            if property_type == "Nhà phố":
                city_mode = st.radio("Khu vực", ["Hồ Chí Minh", "Hà Nội"], horizontal=True)
            else:
                city_mode = "All"
            # ... (Code cũ phần city_mode bên trên giữ nguyên) ...

            # === [PHẦN MỚI] BẮT ĐẦU CHÈN TỪ ĐÂY ===
            st.markdown("#### 📍 Xác định vị trí")
            
            # 1. Khởi tạo Session State
            if 'lat_val' not in st.session_state: st.session_state.lat_val = 10.7769
            if 'lon_val' not in st.session_state: st.session_state.lon_val = 106.7009

            # 2. Tạo 2 Tab chuyển đổi
            tab_search, tab_manual = st.tabs(["🔍 Tìm theo Địa chỉ", "🛠️ Nhập Tọa độ (Thủ công)"])

            # --- TAB 1: DÀNH CHO NGƯỜI DÙNG PHỔ THÔNG ---
            with tab_search:
                col_s1, col_s2 = st.columns([3, 1])
                with col_s1:
                    address_input = st.text_input("Nhập địa chỉ/tên đường:", placeholder="VD: Landmark 81...", label_visibility="collapsed")
                with col_s2:
                    btn_find = st.button("Tìm", type="primary", use_container_width=True)
                
                if btn_find and address_input:
                    try:
                        # --- SỬA ĐOẠN NÀY ---
                        # Lấy token từ secrets
                        mapbox_key = st.secrets["MAPBOX_TOKEN"] 
                        
                        # Khởi tạo MapBox Geocoder
                        geolocator = MapBox(api_key=mapbox_key)
                        
                        # MapBox tìm rất nhanh, timeout thấp cũng được
                        location = geolocator.geocode(address_input, timeout=10)
                        
                        if location:
                            st.session_state.lat_val = location.latitude
                            st.session_state.lon_val = location.longitude
                            st.success(f"✅ Mapbox tìm thấy: {location.address}")
                            sleep(0.5)
                            st.rerun()
                        else:
                            st.warning("⚠️ Mapbox không tìm thấy địa chỉ này.")
                        # --------------------
                        
                    except Exception as e:
                        st.error(f"Lỗi: {e}. (Kiểm tra lại Token trong Secrets)")

            # --- TAB 2: DÀNH CHO NGƯỜI DÙNG KỸ TÍNH (HIỆN CÁI NÀY LÀ CHUẨN NHẤT) ---
            with tab_manual:
                st.caption("Nhập tọa độ chính xác (Decimal Degrees):")
                c_lat, c_lon = st.columns(2)
                with c_lat:
                    # Input này tự động lấy giá trị từ session_state (do Tab 1 tìm được)
                    # Và nếu người dùng sửa ở đây, nó cũng cập nhật ngược lại session_state
                    lat_manual = st.number_input("Vĩ độ (Lat)", value=st.session_state.lat_val, format="%.5f", key="input_lat_manual")
                with c_lon:
                    lon_manual = st.number_input("Kinh độ (Lon)", value=st.session_state.lon_val, format="%.5f", key="input_lon_manual")
                
                # Cập nhật lại biến Session State nếu người dùng sửa tay
                st.session_state.lat_val = lat_manual
                st.session_state.lon_val = lon_manual
            # === [PHẦN MỚI] KẾT THÚC ===

                # ... (Code bên trong giữ nguyên cho đến phần Vị trí) ...
            # 2. Form nhập liệu (Biến đổi theo property_type)
            with st.form("prediction_form"):
                
                # --- PHẦN 1: DỰ ÁN (Chỉ hiện cho Chung cư) ---
                project_name = "Others" 
                if property_type == "Căn hộ Chung cư":
                    st.markdown("**Thông tin Dự án:**")
                    project_options = loader.get_project_list()
                    project_options.insert(0, "Khác / Chưa xác định")
                    project_name = st.selectbox("Tên dự án", project_options)

                # --- PHẦN 2: THÔNG SỐ KỸ THUẬT ---
                st.markdown("**Thông số kỹ thuật:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    area = st.number_input("Diện tích (m²)", min_value=5.0, value=50.0)
                    
                    # Chung cư thì không cần Mặt tiền & Đường vào
                    if property_type != "Căn hộ Chung cư":
                        front_width = st.number_input("Mặt tiền (m)", min_value=0.0, value=5.0)
                        access_road = st.number_input("Đường (m)", min_value=0.0, value=3.0)
                    else:
                        front_width = 0.0
                        access_road = 0.0
                    
                with col2:
                    # A. SỐ PHÒNG NGỦ (Ẩn với Đất nền)
                    if property_type != "Đất nền":
                        bedrooms = st.number_input("Số PN", min_value=0, value=2)
                    else:
                        bedrooms = 0

                    # B. SỐ TẦNG (Ẩn với Đất nền VÀ Chung cư) <--- CẬP NHẬT Ở ĐÂY
                    if property_type != "Đất nền" and property_type != "Căn hộ Chung cư":
                        floors = st.number_input("Số tầng", min_value=0, value=1)
                    else:
                        floors = 0 # Chung cư mặc định là 0 (hoặc 1 tùy logic model, ở đây gán 0 cho sạch)

                    # C. TOILET (Ẩn với Đất nền)
                    if property_type != "Đất nền":
                        toilet = st.number_input("Toilet", min_value=0, value=1)
                    else:
                        toilet = 0

                st.markdown("---")
                
                # --- PHẦN 3: ĐẶC ĐIỂM ---
                c_legal, c_dir = st.columns(2)
                
                with c_legal:
                    legal = st.selectbox("Pháp lý", ["Sổ hồng/Sổ đỏ", "Hợp đồng mua bán", "Vi bằng/Giấy tay", "Khác"])
                
                with c_dir:
                    # Logic Ẩn/Hiện Hướng nhà
                    should_show_direction = True
                    if property_type == "Căn hộ Chung cư": should_show_direction = False
                    if property_type == "Nhà phố" and city_mode == "Hà Nội": should_show_direction = False
                    
                    if should_show_direction:
                        direction = st.selectbox("Hướng", ["Chưa xác định", "Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc"])
                    else:
                        direction = "Chưa xác định"

                # Nội thất (Ẩn với Đất nền)
                if property_type != "Đất nền":
                    interior = st.selectbox("Nội thất", ["Đầy đủ", "Cơ bản", "Nhà trống", "Thỏa thuận", "Chưa xác định"])
                else:
                    interior = "Chưa xác định"
                
                st.markdown("---")
                st.markdown("**📍 Vị trí đã chọn:**")
                
                # Hiển thị tọa độ đang được lưu trong Session State để người dùng yên tâm
                st.info(f"Kinh độ (Lon): {st.session_state.lon_val:.5f} | Vĩ độ (Lat): {st.session_state.lat_val:.5f}")

                # Nút Submit (Giữ nguyên)
                submit_btn = st.form_submit_button("🚀 Dự báo ngay", type="primary")

            # 3. Đóng gói dữ liệu
            user_inputs = {
                "area": area,
                "front_width": front_width,
                "access_road": access_road,
                "bedrooms": bedrooms,
                "floors": floors,
                "toilet": toilet,
                "legal": legal,
                "direction": direction,
                "interior": interior,
                "project_name": project_name,
                "lat": st.session_state.lat_val,
                "lon": st.session_state.lon_val
            }
            
            return nav_mode, user_inputs, city_mode, property_type, submit_btn