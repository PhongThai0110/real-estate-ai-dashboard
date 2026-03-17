import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import altair as alt
# ==============================================================================
# 1. CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS)
# ==============================================================================
# FILE: views/dashboard.py
# Thay thế đoạn DARK_THEME_LAYOUT cũ bằng đoạn này:

DARK_THEME_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#ffffff'), # Ép toàn bộ font chữ cơ bản thành TRẮNG TINH
    
    # Cấu hình cụ thể cho Tiêu đề
    #title=dict(
     #   font=dict(color='#38bdf8', size=18) # Màu xanh neon nổi bật cho tiêu đề biểu đồ
    #),
    
    # Cấu hình trục X (Trục ngang)
    xaxis=dict(
        title_font=dict(color='#e2e8f0'), # Màu chữ tiêu đề trục (VD: Giá TB)
        tickfont=dict(color='#cbd5e1'),   # Màu chữ các con số trên trục
        gridcolor='rgba(255, 255, 255, 0.1)', # Lưới mờ
        showgrid=True
    ),
    
    # Cấu hình trục Y (Trục dọc)
    yaxis=dict(
        title_font=dict(color='#e2e8f0'),
        tickfont=dict(color='#cbd5e1'),
        gridcolor='rgba(255, 255, 255, 0.1)',
        showgrid=True
    ),
    
    # Cấu hình Chú thích (Legend)
    legend=dict(
        font=dict(color='#e2e8f0'),
        bgcolor='rgba(0,0,0,0)'
    ),
    
    margin=dict(t=40, l=10, r=10, b=10),
)
def filter_smart_coordinates(df, city_mode="All"):
    """
    [NÂNG CẤP V2] Lọc toạ độ dựa trên lựa chọn của người dùng (city_mode).
    
    Args:
        df: DataFrame chứa cột 'lat', 'lon'.
        city_mode: Giá trị lấy từ Sidebar ("Hồ Chí Minh", "Hà Nội", hoặc "All").
    """
    if df is None or df.empty: return df
    
    # 1. Copy data để không ảnh hưởng gốc
    df_clean = df.copy()
    
    # 2. Kiểm tra cột tồn tại
    if 'lat' not in df_clean.columns or 'lon' not in df_clean.columns:
        return df_clean

    # 3. Lọc Rác cơ bản (Bắt buộc phải làm)
    # Loại bỏ NaN
    df_clean = df_clean.dropna(subset=['lat', 'lon'])
    # Loại bỏ toạ độ 0.0 (Phi Châu)
    df_clean = df_clean[(df_clean['lat'] != 0) & (df_clean['lon'] != 0)]
    # Loại bỏ toạ độ ngoài lãnh thổ VN (Sơ bộ)
    df_clean = df_clean[(df_clean['lat'] > 8.0) & (df_clean['lat'] < 24.0)]
    df_clean = df_clean[(df_clean['lon'] > 102.0) & (df_clean['lon'] < 110.0)]

    if df_clean.empty: return df_clean

    # 4. [LOGIC MỚI] Lọc theo City Mode
    # Lấy vĩ độ 16.0 (Đà Nẵng/Đèo Hải Vân) làm ranh giới Bắc - Nam tự nhiên
    
    if city_mode == "Hồ Chí Minh":
        # Chỉ lấy Miền Nam (Vĩ độ < 16)
        df_clean = df_clean[df_clean['lat'] < 16.0]
        
    elif city_mode == "Hà Nội":
        # Chỉ lấy Miền Bắc (Vĩ độ >= 16)
        df_clean = df_clean[df_clean['lat'] >= 16.0]
        
    else:
        # Trường hợp "All" (Dành cho Chung cư/Đất nền hoặc khi chưa chọn khu vực)
        # -> KHÔNG LỌC GÌ CẢ (Giữ lại cả Bắc và Nam để hiển thị hết)
        pass 

    return df_clean

def format_price(val):
    if val >= 1: return f"{val:.2f} Tỷ"
    return f"{val*1000:.0f} Tr"


def get_active_selections(chart_key):
    """Hàm trích xuất dữ liệu selection từ st.session_state cho mọi loại biểu đồ"""
    if chart_key in st.session_state:
        selection = st.session_state[chart_key].get("selection", {})
        points = selection.get("points", [])
        if len(points) > 0:
            return points # Trả về toàn bộ list các điểm được chọn
    return None
# ==============================================================================
# 2. CÁC BIỂU ĐỒ CHÍNH (CHARTS)
# ==============================================================================

def chart_heatmap_location(df,city_mode="All"):
    """
    Bản đồ phân bố Bất động sản (Sử dụng Mapbox).
    """
    # 1. Lấy Token từ Secrets (BẢO MẬT)
    mapbox_token = None
    try:
        # Cố gắng lấy token từ cấu hình của Streamlit
        mapbox_token = st.secrets["MAPBOX_TOKEN"]
    except (FileNotFoundError, KeyError):
        # Nếu chưa cấu hình secrets thì báo lỗi và dừng lại
        st.error("⚠️ Chưa cấu hình MAPBOX_TOKEN trong .streamlit/secrets.toml (local) hoặc Settings/Secrets (Cloud).")
        return None

    # 2. Cài đặt Token cho Plotly Express
    # Dòng này bắt buộc phải có trước khi vẽ mapbox style "xịn"
    px.set_mapbox_access_token(mapbox_token)

    # ==================================================
    # Xử lý dữ liệu (Giữ nguyên code cũ của bạn)
    # ==================================================
    clean_df = filter_smart_coordinates(df,city_mode=city_mode)
    
    if clean_df is None or clean_df.empty:
        st.warning("⚠️ Không có dữ liệu toạ độ hợp lệ.")
        return None

    hover_name = 'district'
    if 'project_name_raw' in clean_df.columns:
        hover_name = 'project_name_raw'
    elif 'Tin_BĐS' not in clean_df.columns:
         clean_df['Tin_BĐS'] = "BĐS #" + clean_df.index.astype(str)
         hover_name = 'Tin_BĐS'
    # ==================================================

    clean_df['_row_id'] = clean_df.index.astype(str) # Đảm bảo ID là string để tránh lỗi khi truyền vào custom_data
    try:
        fig = px.scatter_mapbox(
            clean_df,
            lat="lat",
            lon="lon",
            color="price",
            size="area",
            hover_name=hover_name,
            custom_data=['_row_id'], # Giữ lại ID gốc để truy xuất khi chọn điểm
            labels={
                "price": "Giá (Tỷ)", 
                "area": "Diện tích (m²)"
            },
            hover_data={
                "price": ":.2f", # Giữ 2 số thập phân (VD: 9.99)
                "area": ":.1f",  # Giữ 1 số thập phân (VD: 400.0)
                "lat": False,    # Tắt hiển thị vĩ độ rườm rà
                "lon": False     # Tắt hiển thị kinh độ rườm rà
            },
            size_max=15,
            zoom=10,
            # Giữ nguyên dải màu của bạn, nó khá hợp với nền tối
            color_continuous_scale=[
                (0.0, '#0f172a'), 
                (0.5, '#0ea5e9'), 
                (1.0, '#ffffff') 
            ],
  
            # --- [THAY ĐỔI QUAN TRỌNG Ở ĐÂY] ---
            # Code cũ: mapbox_style="carto-darkmatter",
            # Code mới: Sử dụng Mapbox Style URL chính chủ.
            # Các lựa chọn style tối đẹp:
            # 1. "mapbox://styles/mapbox/dark-v11" (Tối tiêu chuẩn, sạch sẽ)
            # 2. "mapbox://styles/mapbox/navigation-night-v1" (Tối kiểu bản đồ dẫn đường xe hơi - Rất ngầu)
            mapbox_style="mapbox://styles/mapbox/dark-v11", 
            # -----------------------------------

            height=500,
            #title=f":material/location_on: Bản đồ phân bố ({len(clean_df)} tin)"
        )
        
        # Áp dụng dark theme layout chung của bạn (nếu có biến này)
        # Nếu chưa có biến này, hãy đảm bảo fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        if 'DARK_THEME_LAYOUT' in globals():
             fig.update_layout(**DARK_THEME_LAYOUT)
        else:
             # Fallback nếu không tìm thấy biến global DARK_THEME_LAYOUT
             fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(t=40, l=10, r=10, b=10)
             )

        return fig

    except Exception as e:
        # Bắt lỗi cụ thể nếu liên quan đến token
        err_msg = str(e).lower()
        if "mapbox access token" in err_msg or "401" in err_msg:
             st.error("Lỗi xác thực Mapbox: Token không hợp lệ hoặc chưa được cài đặt đúng.")
        else:
             st.error(f"Lỗi vẽ bản đồ: {e}")
        return None


def chart_top_expensive_projects(df, city_mode="All",key_bar="top_bar"):
    """
    Top Khu vực/Dự án Đắt đỏ (Biểu đồ + Bản đồ Mapbox).
    Đã thêm bộ lọc chống nhiễu (Chỉ xếp hạng những nơi có đủ số lượng tin)
    """
    if df is None or df.empty: return

    # --- [BƯỚC 1: CÀI ĐẶT MAPBOX] ---
    try:
        mapbox_token = st.secrets["MAPBOX_TOKEN"]
        px.set_mapbox_access_token(mapbox_token)
    except:
        st.warning("⚠️ Chưa có Mapbox Token. Bản đồ có thể không hiển thị đúng style.")
    # --------------------------------

    group_col = None
    label_title = ""
    
    if 'project_name_raw' in df.columns:
        group_col = 'project_name_raw'; label_title = "Dự án"
    elif 'district_mapped' in df.columns:
        group_col = 'district_mapped'; label_title = "Quận/Huyện"
        df_clean = df.dropna(subset=[group_col]).copy()
    elif 'geo_cluster' in df.columns:
        group_col = 'geo_cluster'; label_title = "Khu vực"
    
    if not group_col:
        st.info("Không đủ thông tin để xếp hạng.")
        return

    # Lọc toạ độ trước khi tính toán
    df_clean = filter_smart_coordinates(df, city_mode=city_mode)
    
    # Gom nhóm và tính toán
    stats = df_clean.groupby(group_col).agg({
        'price': 'mean',
        'lat': 'mean',
        'lon': 'mean',
        'area': 'count' # Mượn cột area để đếm số lượng tin đăng
    }).reset_index()
    
    # --- [SỬA LỖI LOGIC Ở ĐÂY] ---
    # THIẾT LẬP NGƯỠNG TỐI THIỂU ĐỂ ĐƯỢC LÊN BẢNG XẾP HẠNG
    # Đảm bảo dữ liệu đủ lớn để giá trị trung bình phản ánh đúng thực tế
    MIN_LISTINGS = 15 
    stats = stats[stats['area'] >= MIN_LISTINGS] 
    # -----------------------------
    
    top_10 = stats.sort_values(by='price', ascending=False).head(10)
    
    if top_10.empty:
        st.info(f"Chưa đủ dữ liệu để xếp hạng Top 10 (cần khu vực có trên {MIN_LISTINGS} tin đăng).")
        return

    st.subheader(f":material/diamond: Top 10 {label_title} Đắt đỏ nhất")

    c1, c2 = st.columns([1, 1])
    
    # --- CỘT TRÁI: BIỂU ĐỒ CỘT ---
    with c1:
        fig_bar = px.bar(
            top_10,
            x='price',
            y=group_col,
            orientation='h',
            color='price',
            color_continuous_scale='Viridis', 
            text_auto='.2s',
            labels={'price': 'Giá TB (Tỷ)', group_col: label_title},
            title="Xếp hạng theo giá"
        )
        
        # Áp dụng Dark Theme
        if 'DARK_THEME_LAYOUT' in globals():
             fig_bar.update_layout(**DARK_THEME_LAYOUT)
        
        fig_bar.update_xaxes(showgrid=False)
        st.plotly_chart(fig_bar, width="stretch", key=key_bar, on_select="rerun", selection_mode="points")
        
    # --- CỘT PHẢI: BẢN ĐỒ MAPBOX ---
    with c2:
        st.markdown(f"**:material/map: Vị trí thực tế:**")
        
        fig_map = px.scatter_mapbox(
            top_10,
            lat="lat",
            lon="lon",
            color="price",
            size="price", # Bong bóng to nhỏ tùy theo giá
            hover_name=group_col,
            
            # Việt hóa tooltip
            labels={
                "price": "Giá trung bình (Tỷ)",
                group_col: "Khu vực"
            },
            hover_data={
                "price": ":.1f",  
                "lat": False,     
                "lon": False      
            },
            
            color_continuous_scale='Viridis',
            zoom=10, 
            mapbox_style="mapbox://styles/mapbox/navigation-night-v1",
            height=400
        )
        
        # Áp dụng Dark Theme
        if 'DARK_THEME_LAYOUT' in globals():
             fig_map.update_layout(**DARK_THEME_LAYOUT)
        
        st.plotly_chart(fig_map, width="stretch", config={'scrollZoom': True})

def chart_donut_legal(df):
    """Biểu đồ tròn Altair: Legend nằm ngang bên phải (Side Legend)"""
    if 'legal' not in df.columns: return None
    
    # 1. Thống kê và tính phần trăm
    legal_counts = df['legal'].fillna("Chưa xác định").value_counts().reset_index()
    legal_counts.columns = ['Pháp_lý', 'Số_lượng']
    
    total = legal_counts['Số_lượng'].sum()
    legal_counts['Tỷ_lệ'] = (legal_counts['Số_lượng'] / total * 100).round(1)
    
    legal_counts['PhapLy_HienThi'] = legal_counts['Pháp_lý'] + " (" + legal_counts['Tỷ_lệ'].astype(str) + "%)"
    
    color_map = {
        "Sổ hồng/Sổ đỏ": "#38bdf8", 
        "Hợp đồng mua bán": "#818cf8",
        "Hợp đồng mua bán/Chờ sổ": "#c084fc", 
        "Vi bằng/Giấy tay": "#f43f5e", 
        "Giấy tờ khác": "#fbbf24", 
        "Chưa xác định": "#94a3b8"
    }
    domain = legal_counts['PhapLy_HienThi'].tolist()
    range_ = [color_map.get(name, "#94a3b8") for name in legal_counts['Pháp_lý']]
    
    click_selection = alt.selection_point(fields=['Pháp_lý'], name='legal_click')
    
    # 2. VẼ BIỂU ĐỒ 
    donut = alt.Chart(legal_counts).mark_arc(
        innerRadius=65,   # Bóp nhỏ xíu nữa để nhường không gian bề ngang cho Legend
        outerRadius=115,  
        stroke="#0f172a", strokeWidth=1.5
    ).encode(
        theta=alt.Theta(field="Số_lượng", type="quantitative"),
        color=alt.Color(
            field="PhapLy_HienThi",
            type="nominal", 
            scale=alt.Scale(domain=domain, range=range_),
            legend=alt.Legend(
                title=None, 
                
                # --- [CHÌA KHÓA Ở ĐÂY: CHUYỂN SANG BÊN PHẢI] ---
                orient="right",    # Đưa toàn bộ khối chú thích sang bên phải cục bánh
                
                columns=1,         # Giữ nguyên xếp thành 1 hàng dọc
                labelFontSize=13, 
                labelColor="#cbd5e1", 
                symbolType="circle",
                labelLimit=0, 
                offset=15,         # Cách cụm bánh 15px
                rowPadding=8       # Tăng nhẹ khoảng cách các dòng cho thoáng mắt
            )
        ),
        tooltip=[
            alt.Tooltip('Pháp_lý', title='Pháp lý'),
            alt.Tooltip('Số_lượng', title='Số lượng'),
            alt.Tooltip('Tỷ_lệ', title='Tỷ lệ (%)')
        ],
        opacity=alt.condition(click_selection, alt.value(1.0), alt.value(0.3))
    ).add_params(click_selection).properties(
        
        # Cân bằng lại Padding (xả bớt lề trên và dưới vì chữ đã dọn sang nhà bên phải)
        padding={"left": 10, "top": 20, "right": 20, "bottom": 20} 
    )
    
    chart = donut.configure_view(strokeWidth=0).configure(background='transparent')
    
    return chart
def chart_scatter_area_price(df):
    """Biểu đồ tương quan Diện tích - Giá (Đã tối ưu màu sắc High-Contrast)"""
    if df is None or df.empty: return None
    
    df_zoom = df[(df['area'] > 0) & (df['price'] > 0)].copy()
    if df_zoom['area'].max() > 1000:
        df_zoom = df_zoom[df_zoom['area'] < 1000]
        
    # --- BẢNG MÀU THEO NGỮ NGHĨA (SEMANTIC COLORS) ---
    semantic_colors = {
        "Sổ hồng/Sổ đỏ": "#38bdf8",     # Xanh Cyan sáng (An toàn)
        "Hợp đồng mua bán": "#fbbf24",  # Vàng Cam (Chờ đợi)
        "Vi bằng/Giấy tay": "#f87171",   # Đỏ san hô (Rủi ro)
        "Khác": "#c084fc",              # Tím sáng
        "Giấy tờ khác": "#c084fc",      # (Tên dự phòng)
        "Chưa xác định": "#94a3b8"      # Xám
    }
    df_zoom['_row_id'] = df_zoom.index.astype(str) # Đảm bảo ID là string để tránh lỗi khi truyền vào custom_data
    fig = px.scatter(
        df_zoom, x='area', y='price',
        color='legal' if 'legal' in df.columns else None,
        color_discrete_map=semantic_colors, # Ép dùng bảng màu tùy chỉnh
        trendline="ols",
        custom_data=['_row_id'], # Giữ lại ID gốc để truy xuất khi chọn điểm
        labels={'area': 'Diện tích (m²)', 'price': 'Giá (Tỷ)', 'legal': 'Pháp lý'},
        height=500, 
        opacity=0.8 # Tăng nhẹ độ đậm của chấm để rõ màu hơn
    )
    
    # Định dạng lại lưới và nền
    if 'DARK_THEME_LAYOUT' in globals():
        fig.update_layout(**DARK_THEME_LAYOUT)
        
    # Tinh chỉnh lưới mờ và di chuyển chú thích (Legend) cho thoáng
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)') 
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_layout(legend=dict(
        orientation="h", 
        yanchor="bottom", 
        y=1.02, 
        xanchor="right", 
        x=1
    ))
    
    return fig


def chart_box_alley_impact(df):
    """Biểu đồ phân phối giá theo loại đường (Đã chuyển sang Bar Chart cho dễ hiểu)"""
    if 'access_road' not in df.columns: return None
    
    # 1. Lọc dữ liệu sạch
    work_df = df[(df['access_road'] > 0) & (df['area'] > 0)].copy()
    work_df['price_per_m2'] = (work_df['price'] * 1000) / work_df['area']
    work_df = work_df[work_df['price_per_m2'] < 500]
    
    # 2. Phân loại đường
    def classify(w):
        if w < 2.5: return "1. Hẻm xe máy (< 2.5m)"
        elif w < 5.0: return "2. Hẻm 1 ô tô (2.5 - 5m)"
        elif w < 10: return "3. Hẻm 2 ô tô (5 - 10m)"
        return "4. Đường lớn (> 10m)"
        
    work_df['Loai_duong'] = work_df['access_road'].apply(classify)
    
    # 3. Tính toán mức giá phổ biến (Trung vị - Median) thay vì để Box Plot tự vẽ
    stats_df = work_df.groupby('Loai_duong').agg(
        Gia_Phobien=('price_per_m2', 'median')
    ).reset_index()
    
    stats_df = stats_df.sort_values('Loai_duong')
    
    # 4. Vẽ biểu đồ CỘT (Dễ hiểu nhất với đại chúng)
    fig = px.bar(
        stats_df, 
        x='Loai_duong', 
        y='Gia_Phobien', 
        color='Loai_duong',
        text='Gia_Phobien', # Hiển thị số ngay trên cột
        labels={
            'Gia_Phobien': 'Đơn giá phổ biến (Triệu/m²)', 
            'Loai_duong': '' 
        }
    )
    
    # Làm tròn số trên cột và đưa lên phía trên
    fig.update_traces(texttemplate='%{text:.0f} Tr', textposition='outside')
    
    fig.update_layout(showlegend=False) # Tắt chú thích vì trục X đã rõ
    fig.update_yaxes(range=[0, stats_df['Gia_Phobien'].max() * 1.2]) # Tăng khoảng không phía trên để số không bị cắt
    
    if 'DARK_THEME_LAYOUT' in globals():
        fig.update_layout(**DARK_THEME_LAYOUT)
        
    return fig


def chart_bar_shape_classification(df):
    """Phân loại hình dáng đất thành các nhóm thân thiện với BĐS để lọc chéo"""
    if 'front_width' not in df.columns: return None
    
    # 1. Lọc data hợp lệ và tính tỷ lệ (Dài/Rộng)
    work_df = df[(df['front_width'] > 0) & (df['area'] > 0)].copy()
    work_df['shape_ratio'] = (work_df['area'] / work_df['front_width']) / work_df['front_width']
    
    # 2. Phân loại theo thuật ngữ BĐS
    def classify_shape(ratio):
        if ratio < 1.0: return "1. Bề ngang rộng (Mặt tiền > Dài)"
        elif ratio <= 2.5: return "2. Vuông vức (1:1 đến 1:2.5)"
        elif ratio <= 5.0: return "3. Nhà ống (1:2.5 đến 1:5)"
        else: return "4. Siêu hẹp & Dài (> 1:5)"
        
    work_df['Hinh_Dang'] = work_df['shape_ratio'].apply(classify_shape)
    
    # 3. Thống kê số lượng
    stats_df = work_df['Hinh_Dang'].value_counts().reset_index()
    stats_df.columns = ['Hinh_Dang', 'So_Luong']
    stats_df = stats_df.sort_values('Hinh_Dang') # Sắp xếp theo thứ tự 1,2,3,4
    
    # 4. Vẽ biểu đồ Cột (Bar Chart)
    fig = px.bar(
        stats_df, x='Hinh_Dang', y='So_Luong', color='Hinh_Dang',
        text='So_Luong',
        labels={'Hinh_Dang': '', 'So_Luong': 'Số lượng tin'},
        # Dùng bảng màu gradient từ xanh dương sang tím cho đẹp
        color_discrete_sequence=['#38bdf8', '#818cf8', '#c084fc', '#f472b6'] 
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, yaxis_range=[0, stats_df['So_Luong'].max() * 1.2])
    
    if 'DARK_THEME_LAYOUT' in globals():
        fig.update_layout(**DARK_THEME_LAYOUT)
        
    return fig

# ==============================================================================
# 3. KPI METRICS
# ==============================================================================

def render_kpi_metrics(df):
    if df is None or df.empty: return
    
    valid = df[(df['price'] > 0) & (df['area'] > 0)].copy()
    if valid.empty: return
    
    avg_price = valid['price'].mean()
    avg_area = valid['area'].mean()
    valid['don_gia'] = (valid['price'] * 1000) / valid['area']
    avg_don_gia = valid[valid['don_gia'] < 1000]['don_gia'].mean()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tin đăng", f"{len(df):,}")
    c2.metric("Giá Rao TB", format_price(avg_price))
    c3.metric("Đơn giá TB", f"{avg_don_gia:,.1f} Tr/m²")
    c4.metric("Diện tích TB", f"{avg_area:,.1f} m²")

# ==============================================================================
# 4. GIAO DIỆN CHÍNH (MAIN UI)
# ==============================================================================


def show_dashboard_ui(df, category_name, city_mode="All"):
    if df is None or df.empty:
        st.warning(f"⚠️ Chưa có dữ liệu cho danh mục: **{category_name}**")
        return

    # =========================================================
    # 🧠 BỘ NÃO V3: KÉT SẮT LƯU TRỮ BỘ LỌC CHỒNG
    # =========================================================
    if "global_filters" not in st.session_state:
        st.session_state.global_filters = {}
    # Bắt tín hiệu từ biểu đồ Pháp lý (Altair)
    legal_altair_state = st.session_state.get("legal_donut", {}).get("selection", {})
    if "legal_click" in legal_altair_state and len(legal_altair_state["legal_click"]) > 0:
        # Lấy tên gốc (VD: "Sổ hồng/Sổ đỏ")
        selected_legal = legal_altair_state["legal_click"][0]["Pháp_lý"] 
        st.session_state.global_filters['Pháp lý'] = selected_legal
    # Đọc tín hiệu từ các biểu đồ Plotly
    alley_pts = get_active_selections("alley_bar")
    if alley_pts and 'x' in alley_pts[0] and 'access_road' in df.columns:
        st.session_state.global_filters['Loại đường'] = alley_pts[0]['x']

    top_pts = get_active_selections("top_bar")
    if top_pts and 'y' in top_pts[0]:
        st.session_state.global_filters['Khu vực'] = top_pts[0]['y']

    map_pts = get_active_selections("main_map")
    if map_pts:
        indices = [p['customdata'][0] for p in map_pts if 'customdata' in p]
        if indices: st.session_state.global_filters['Khoanh vùng Bản đồ'] = indices

    scatter_pts = get_active_selections("scatter_chart")
    if scatter_pts:
        indices = [p['customdata'][0] for p in scatter_pts if 'customdata' in p]
        if indices: st.session_state.global_filters['Khoanh vùng Biểu đồ Giá'] = indices
    
    shape_pts = get_active_selections("shape_bar")
    if shape_pts and 'x' in shape_pts[0] and 'front_width' in df.columns:
        st.session_state.global_filters['Hình dáng'] = shape_pts[0]['x']


    # =========================================================
    # 🎯 THANH CÔNG CỤ XÓA BỘ LỌC TỪNG PHẦN
    # =========================================================
    filters = st.session_state.global_filters
    widget_keys = {
        'Pháp lý': 'legal_donut', 'Loại đường': 'alley_bar', 'Khu vực': 'top_bar',
        'Khoanh vùng Bản đồ': 'main_map', 'Khoanh vùng Biểu đồ Giá': 'scatter_chart','Hình dáng': 'shape_bar'
    }

    if filters:
        st.markdown("**:material/filter_alt: Đang lọc theo TẤT CẢ các điều kiện sau:**")
        cols = st.columns(len(filters) + 1) 
        
        for i, (key, val) in enumerate(list(filters.items())):
            display_val = f"{len(val)} căn" if isinstance(val, list) else str(val)
            with cols[i]:
                if st.button(f"{key}: {display_val[:15]}", key=f"clear_{key}",type="tertiary", icon=":material/close:"):
                    del st.session_state.global_filters[key]
                    w_key = widget_keys.get(key)
                    if w_key and w_key in st.session_state: del st.session_state[w_key]
                    st.rerun()

        with cols[-1]:
            if st.button("XÓA TẤT CẢ", type="secondary", icon=":material/delete_forever:"):
                st.session_state.global_filters = {}
                for w_key in widget_keys.values():
                    if w_key in st.session_state: del st.session_state[w_key]
                st.rerun()
        st.markdown("---")

    # =========================================================
    # ⚙️ HÀM BỘ LỌC ĐỘNG (CHỐNG LỖI SẬP BIỂU ĐỒ TRÒN)
    # =========================================================
    def apply_filters(data, current_filters, ignore_keys=[]):
        """Hàm lọc Dataframe thông minh, cho phép bỏ qua một số key nhất định"""
        d = data.copy()
        for k, v in current_filters.items():
            if k in ignore_keys: 
                continue 
                
            if k == 'Pháp lý':
                d = d[d['legal'] == v]
            elif k == 'Loại đường':
                if "Hẻm xe máy" in v: d = d[d['access_road'] < 2.5]
                elif "Hẻm 1 ô" in v: d = d[(d['access_road'] >= 2.5) & (d['access_road'] < 5.0)]
                elif "Hẻm 2 ô" in v: d = d[(d['access_road'] >= 5.0) & (d['access_road'] < 10.0)]
                elif "Đường lớn" in v: d = d[d['access_road'] >= 10.0]
                
            # --- [SỬA Ở ĐÂY: LOGIC DÒ TÌM THÔNG MINH CHO KHU VỰC/DỰ ÁN] ---
            elif k == 'Khu vực':
                is_filtered = False
                
                # 1. Ưu tiên quét trong cột Tên Dự án trước (Dành cho Chung Cư)
                for proj_col in ['project_name', 'project_name_raw', 'Tên Dự Án']: # Thay bằng tên cột thực tế của bạn nếu khác
                    if proj_col in d.columns and v in d[proj_col].unique():
                        d = d[d[proj_col] == v]
                        is_filtered = True
                        break
                
                # 2. Nếu không tìm thấy trong cột Dự án, quét sang cột Quận/Huyện (Dành cho Nhà Phố)
                if not is_filtered:
                    for dist_col in ['district_mapped', 'district', 'Quận']:
                        if dist_col in d.columns:
                            d = d[d[dist_col] == v]
                            break
            # -------------------------------------------------------------
            
            elif k in ['Khoanh vùng Bản đồ', 'Khoanh vùng Biểu đồ Giá']:
                unique_indices_str = [str(x) for x in set(v)]
                d = d[d.index.astype(str).isin(unique_indices_str)]
            elif k == 'Hình dáng':
                ratio = (d['area'] / d['front_width']) / d['front_width']
                if "Bề ngang rộng" in v: d = d[ratio < 1.0]
                elif "Vuông vức" in v: d = d[(ratio >= 1.0) & (ratio <= 2.5)]
                elif "Nhà ống" in v: d = d[(ratio > 2.5) & (ratio <= 5.0)]
                elif "Siêu hẹp" in v: d = d[ratio > 5.0]
        return d

    # Tạo 2 luồng Dataframe
    df_filtered = apply_filters(df, filters) 
    df_for_legal = apply_filters(df, filters, ignore_keys=['Pháp lý']) # Luồng riêng biệt chống sập bánh

    if df_filtered.empty:
        st.error("Không có căn nhà nào thỏa mãn ĐỒNG THỜI tất cả các điều kiện trên! Vui lòng xóa bớt bộ lọc.", icon=":material/search_off:")
        return

    # =========================================================
    # 🎨 VẼ GIAO DIỆN
    # =========================================================
    render_kpi_metrics(df_filtered)
    st.markdown("---")

    st.subheader(f":material/location_on: Bản đồ phân bố ({len(df_filtered)} tin)")
    fig_map = chart_heatmap_location(df_filtered, city_mode=city_mode)
    if fig_map: 
        st.plotly_chart(fig_map, width="stretch", config={'scrollZoom': True}, key="main_map", on_select="rerun", selection_mode="points")
    st.markdown("---")

    chart_top_expensive_projects(df_filtered, city_mode=city_mode, key_bar="top_bar")
    st.markdown("---")

    # 4. PHÂN TÍCH SÂU
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f":material/balance: Cơ cấu Pháp lý")
        
        # Dùng df_for_legal để biểu đồ không bị sập khi đang lọc
        fig_donut = chart_donut_legal(df_for_legal)
        if fig_donut: 
            st.altair_chart(
                fig_donut, 
                width='stretch', 
                key="legal_donut", 
                on_select="rerun" # Kích hoạt tính năng Native Streamlit Cross-filtering
            )

    with c2:
        st.subheader(":material/trending_up: Xu hướng Diện tích - Giá")
        fig_scatter = chart_scatter_area_price(df_filtered)
        if fig_scatter: st.plotly_chart(fig_scatter, width="stretch", key="scatter_chart", on_select="rerun", selection_mode="points")

    if 'access_road' in df_filtered.columns:
        st.markdown("---")
        st.subheader(":material/add_road: Phân phối giá theo loại đường")
        fig_box = chart_box_alley_impact(df_filtered)
        if fig_box: st.plotly_chart(fig_box, width="stretch", key="alley_bar", on_select="rerun", selection_mode="points")
        
    if 'front_width' in df_filtered.columns:
        st.subheader(":material/architecture: Phân loại Hình dáng đất")
        fig_shape = chart_bar_shape_classification(df_filtered)
        if fig_shape: 
            st.plotly_chart(fig_shape, width="stretch", key="shape_bar", on_select="rerun", selection_mode="points")