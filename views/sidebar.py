import streamlit as st
import os
from src import loader
from geopy.geocoders import MapBox
from time import sleep


def update_lat():
    """Chỉ chạy khi người dùng gõ tay vào ô Vĩ độ"""
    st.session_state.lat_val = st.session_state.input_lat_manual

def update_lon():
    """Chỉ chạy khi người dùng gõ tay vào ô Kinh độ"""
    st.session_state.lon_val = st.session_state.input_lon_manual
def show_sidebar():
    with st.sidebar:
        # 1. LOGO
        if os.path.exists("assets/logo_fpt.png"):
            st.image("assets/logo_fpt.png", width=150)  
        st.header(":material/domain: Real Estate AI")
        
        # 2. MENU
        nav_mode = st.radio("Chế độ:", [":material/search: Dự báo giá nhà", 
                ":material/dashboard: Dashboard Phân tích",
                ":material/chat: Chatbot Tư vấn"])
        st.markdown("---")
        
        # === A. CHẾ ĐỘ DASHBOARD ===
        if nav_mode == ":material/dashboard: Dashboard Phân tích":
            st.subheader("Tùy chọn hiển thị")
            dashboard_category = st.selectbox(
                "Chọn dữ liệu:", 
                ["Nhà phố Hồ Chí Minh", "Nhà phố Hà Nội", "Căn hộ Chung cư", "Đất nền", "Biệt thự / Villa"]
            )
          
            return nav_mode, {}, dashboard_category, "All", False

        # === B. CHẾ ĐỘ CHATBOT  ===
        elif nav_mode == ":material/chat: Chatbot Tư vấn":
            # Khi chọn Chatbot, ta không cần hiển thị form nhập liệu phức tạp
            # Chỉ cần hiển thị thông tin tác giả hoặc để trống cho sạch
            st.info("🤖 **Trợ lý ảo Real Estate AI**\n\nSẵn sàng hỗ trợ bạn 24/7.")
            
            # Trả về các giá trị rỗng để app.py không bị lỗi
            return nav_mode, {}, "All", "Chatbot", False

        # === C. CHẾ ĐỘ DỰ BÁO ===
        else:
            st.subheader(":material/edit_document: Nhập thông tin BĐS")
            
            # 1. Chọn Loại hình & Khu vực trước
            property_type = st.selectbox("Loại hình BĐS", ["Nhà phố", "Căn hộ Chung cư", "Đất nền", "Biệt thự / Villa"])
            
            if property_type == "Nhà phố":
                city_mode = st.radio("Khu vực", ["Hồ Chí Minh", "Hà Nội"], horizontal=True)
            else:
                city_mode = "All"

            # ------------------------------------------------------------------
            # PHẦN VỊ TRÍ (GIỮ NGUYÊN CODE CỦA BẠN)
            st.markdown("#### :material/location_on: Xác định vị trí ")
            if 'lat_val' not in st.session_state: st.session_state.lat_val = 10.7769
            if 'lon_val' not in st.session_state: st.session_state.lon_val = 106.7009

            tab_search, tab_manual = st.tabs([":material/search: Tìm theo Địa chỉ", ":material/my_location: Nhập Tọa độ (Thủ công)"])

            with tab_search:
                col_s1, col_s2 = st.columns([3, 1])
                with col_s1:
                    address_input = st.text_input("Nhập địa chỉ/tên đường:", placeholder="VD: Landmark 81...", label_visibility="collapsed")
                with col_s2:
                    btn_find = st.button("Tìm", type="primary", use_container_width=True)
                
                if btn_find and address_input:
                    try:
                        mapbox_key = st.secrets["MAPBOX_TOKEN"] 
                        geolocator = MapBox(api_key=mapbox_key)
                        location = geolocator.geocode(address_input, timeout=10)
                        
                        if location:
                            # 1. Cập nhật biến gốc (như bạn đã làm)
                            st.session_state.lat_val = location.latitude
                            st.session_state.lon_val = location.longitude

                            # 2. [THÊM MỚI] Cập nhật luôn biến của Widget nhập tay
                            # Giúp ô nhập liệu ở Tab 2 đổi số ngay lập tức
                            st.session_state.input_lat_manual = location.latitude
                            st.session_state.input_lon_manual = location.longitude
                            
                            st.success(f"✅ Mapbox tìm thấy: {location.address}")
                            sleep(0.5)
                            st.rerun()
                        else:
                            st.warning("⚠️ Mapbox không tìm thấy địa chỉ này.")
                    except Exception as e:
                        st.error(f"Lỗi: {e}. (Kiểm tra lại Token trong Secrets)")

            with tab_manual:
                st.info(
                    ":material/tips_and_updates: **Mẹo:** AI sẽ dự báo chuẩn hơn nếu bạn nhập Tọa độ chính xác từng mét "
                    "(thay vì chỉ tìm tên đường chung chung)."
                )

                # 2. Hướng dẫn lấy tọa độ (Thu gọn cho gọn giao diện)
                with st.expander(":material/info: Hướng dẫn lấy tọa độ từ Google Maps"):
                    st.markdown("""
                        1. Truy cập [Google Maps](https://www.google.com/maps).
                        2. Tìm và zoom tới vị trí chính xác của BĐS.
                        3. **Nhấp chuột phải** vào điểm đó.
                        4. Nhấp chuột trái vào dòng số đầu tiên để copy.
                           *(Ví dụ: 10.7769, 106.7009)*
                        5. Paste (Dán) vào 2 ô bên dưới.
                    """)
                
                st.markdown("---")
                # ==========================================================

                st.caption("Nhập tọa độ chính xác (Decimal Degrees):")
                c_lat, c_lon = st.columns(2)
                with c_lat:
                    # [QUAN TRỌNG]
                    # - value lấy từ session (để Mapbox cập nhật được vào đây)
                    # - on_change gọi hàm update (để khi gõ tay thì session mới đổi)
                    # - key phải khớp với tên biến trong hàm update
                    st.number_input(
                        "Vĩ độ (Lat)", 
                        value=st.session_state.lat_val, 
                        format="%.5f", 
                        key="input_lat_manual",  # Key này quan trọng
                        on_change=update_lat     # Gọi hàm khi gõ
                    )

                with c_lon:
                    st.number_input(
                        "Kinh độ (Lon)", 
                        value=st.session_state.lon_val, 
                        format="%.5f", 
                        key="input_lon_manual",  # Key này quan trọng
                        on_change=update_lon     # Gọi hàm khi gõ
                    )
                    
            # ------------------------------------------------------------------

            # 2. Form nhập liệu
            with st.form("prediction_form"):
                
                # --- PHẦN 1: DỰ ÁN (Chỉ hiện cho Chung cư) ---
                project_name = "Others" 
                if property_type == "Căn hộ Chung cư":
                    st.markdown("**Thông tin Dự án:**")
                    project_options = loader.get_project_list()
                    project_options.insert(0, "Khác / Chưa xác định")
                    project_name = st.selectbox("Tên dự án", project_options)

                # --- PHẦN 2: THÔNG SỐ KỸ THUẬT ---
                # --- PHẦN 2: THÔNG SỐ KỸ THUẬT ---
                st.markdown("**Thông số kỹ thuật:**")
                col1, col2 = st.columns(2)
                
                is_corner = 0 

                with col1:
                    # [ĐÃ SỬA] Bỏ tham số 'step' để cho phép nhập tự do như File 1
                    # Vẫn giữ format="%.2f" để hiển thị 2 số lẻ cho đẹp
                    area = st.number_input(
                        "Diện tích (m²)", 
                        min_value=5.0, 
                        value=50.0, 
                        # step=0.1, <--- ĐÃ XÓA DÒNG NÀY ĐỂ MỞ KHÓA BÀN PHÍM
                        format="%.2f", 
                        key="input_area"
                    )
                    
                    if property_type != "Căn hộ Chung cư":
                        front_width = st.number_input(
                            "Mặt tiền (m)", 
                            min_value=0.0, 
                            value=5.0, 
                            # step=0.1, <--- ĐÃ XÓA
                            format="%.2f", 
                            key="input_front"
                        )
                        access_road = st.number_input(
                            "Đường (m)", 
                            min_value=0.0, 
                            value=3.0, 
                            # step=0.1, <--- ĐÃ XÓA
                            format="%.2f", 
                            key="input_road"
                        )
                    else:
                        front_width = 0.0
                        access_road = 0.0
                    
                with col2:
                    if property_type != "Đất nền":
                        # [GIỮ NGUYÊN] Với số nguyên (PN, Tầng...) thì CẦN step=1
                        bedrooms = st.number_input(
                            "Số PN", 
                            min_value=0, 
                            value=2, 
                            step=1, # Giữ lại để bấm +/- cho tiện
                            format="%d", 
                            key="input_bedrooms"
                        )
                    else:
                        bedrooms = 0

                    if property_type != "Đất nền" and property_type != "Căn hộ Chung cư":
                        floors = st.number_input(
                            "Số tầng", 
                            min_value=0, 
                            value=1, 
                            step=1, 
                            format="%d", 
                            key="input_floors"
                        )
                    else:
                        floors = 0

                    if property_type != "Đất nền":
                        toilet = st.number_input(
                            "Toilet", 
                            min_value=0, 
                            value=1, 
                            step=1, 
                            format="%d", 
                            key="input_toilet"
                        )
                    else:
                        toilet = 0

                st.markdown("---")
                
                # --- PHẦN 3: ĐẶC ĐIỂM ---
                c_legal, c_dir = st.columns(2)
                
                with c_legal:
                    legal = st.selectbox("Pháp lý", ["Sổ hồng/Sổ đỏ", "Hợp đồng mua bán", "Vi bằng/Giấy tay", "Khác"])
                
                with c_dir:
                    should_show_direction = True
                    if property_type == "Căn hộ Chung cư": should_show_direction = False
                    if property_type == "Nhà phố" and city_mode == "Hà Nội": should_show_direction = False
                    
                    if should_show_direction:
                        direction = st.selectbox("Hướng", ["Chưa xác định", "Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc"])
                    else:
                        direction = "Chưa xác định"

                # --- TÌM ĐOẠN NÀY ĐỂ THAY THẾ ---
                # (Đoạn cũ chỉ có interior, giờ ta gộp cả interior và is_corner vào đây)
                
                c_int, c_corn = st.columns(2)

                # Cột 1: Nội thất
                with c_int:
                    if property_type != "Đất nền":
                        interior = st.selectbox("Nội thất", ["Đầy đủ", "Cơ bản", "Nhà trống", "Thỏa thuận", "Chưa xác định"])
                    else:
                        interior = "Chưa xác định"
                
                # Cột 2: Lô góc (Chuyển xuống đây)
                with c_corn:
                    if property_type != "Căn hộ Chung cư" and property_type != "Đất nền":  # Chỉ nhà phố và biệt thự mới có lô góc
                        corner_opt = st.selectbox(
                            "Là Lô góc (2 mặt tiền)?", 
                            ["Không", "Có"],
                            index=0,
                            help="Chọn 'Có' nếu BĐS nằm ở góc đường."
                        )
                        is_corner = 1 if corner_opt == "Có" else 0
                    else:
                        is_corner = 0 # Chung cư không có lô góc
                
                st.markdown("---")
                st.markdown("**:material/pin_drop: Vị trí đã chọn:**")
                st.info(f"Kinh độ (Lon): {st.session_state.lon_val:.5f} | Vĩ độ (Lat): {st.session_state.lat_val:.5f}")

                submit_btn = st.form_submit_button(":material/online_prediction: Dự báo ngay", type="primary")

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
                "lon": st.session_state.lon_val,
                "is_corner": is_corner  # <--- [CẬP NHẬT 3] THÊM VÀO DICT TRẢ VỀ
            }
            
            return nav_mode, user_inputs, city_mode, property_type, submit_btn