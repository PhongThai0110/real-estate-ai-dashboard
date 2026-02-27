import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

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


    try:
        fig = px.scatter_mapbox(
            clean_df,
            lat="lat",
            lon="lon",
            color="price",
            size="area",
            hover_name=hover_name,
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


def chart_top_expensive_projects(df, city_mode="All"):
    """
    Top Khu vực/Dự án Đắt đỏ (Biểu đồ + Bản đồ Mapbox).
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
    
    stats = df_clean.groupby(group_col).agg({
        'price': 'mean',
        'lat': 'mean',
        'lon': 'mean',
        'area': 'count'
    }).reset_index()
    
    stats = stats[stats['area'] >= 2] 
    top_10 = stats.sort_values(by='price', ascending=False).head(10)
    
    if top_10.empty:
        st.info("Chưa đủ dữ liệu để xếp hạng Top 10.")
        return

    st.subheader(f":material/diamond: Top 10 {label_title} Đắt đỏ nhất")

    c1, c2 = st.columns([1, 1])
    
    # --- CỘT TRÁI: BIỂU ĐỒ CỘT (Giữ nguyên) ---
    with c1:
        fig_bar = px.bar(
            top_10,
            x='price',
            y=group_col,
            orientation='h',
            color='price',
            color_continuous_scale='Viridis', # Hoặc đổi sang 'Teal' cho hợp tông xanh
            text_auto='.2s',
            labels={'price': 'Giá TB (Tỷ)', group_col: label_title},
            title="Xếp hạng theo giá"
        )
        
        # Áp dụng Dark Theme
        if 'DARK_THEME_LAYOUT' in globals():
             fig_bar.update_layout(**DARK_THEME_LAYOUT)
        
        fig_bar.update_xaxes(showgrid=False)
        st.plotly_chart(fig_bar, width="stretch")
        
    # --- CỘT PHẢI: BẢN ĐỒ MAPBOX (Đã nâng cấp) ---
    with c2:
        st.markdown(f"**:material/map: Vị trí thực tế:**")
        
        fig_map = px.scatter_mapbox(
            top_10,
            lat="lat",
            lon="lon",
            color="price",
            size="price", # Bong bóng to nhỏ tùy theo giá
            hover_name=group_col,
            
            # --- [CẬP NHẬT MỚI: VIỆT HÓA VÀ FORMAT TOOLTIP] ---
            labels={
                "price": "Giá trung bình (Tỷ)",
                group_col: "Khu vực"
            },
            hover_data={
                "price": ":.1f",  # Ép kiểu 1 số thập phân (VD: 30.4)
                "lat": False,     # Tắt hiển thị Vĩ độ (lat)
                "lon": False      # Tắt hiển thị Kinh độ (lon)
            },
            # ---------------------------------------------------
            
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
    """Biểu đồ tròn tỷ lệ Pháp lý"""
    if 'legal' not in df.columns: return None
    
    legal_counts = df['legal'].fillna("Chưa xác định").value_counts().reset_index()
    legal_counts.columns = ['Pháp lý', 'Số lượng']
    
    color_map = {
        "Sổ hồng/Sổ đỏ": "#FF4B4B", "Hợp đồng mua bán": "#1E88E5",
        "Vi bằng/Giấy tay": "#555555", "Giấy tờ khác": "#FFAA00", "Chưa xác định": "#E0E0E0"
    }
    
    fig = px.pie(
        legal_counts, values='Số lượng', names='Pháp lý', 
        hole=0.5, color='Pháp lý', color_discrete_map=color_map,
        #title=f":material/balance: Cơ cấu Pháp lý ({len(df)} tin)"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(**DARK_THEME_LAYOUT)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
    return fig


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
    
    fig = px.scatter(
        df_zoom, x='area', y='price',
        color='legal' if 'legal' in df.columns else None,
        color_discrete_map=semantic_colors, # Ép dùng bảng màu tùy chỉnh
        trendline="ols",
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
        if w < 2.5: return "1. Hẻm nhỏ (< 2.5m)"
        elif w < 5.0: return "2. Hẻm xe hơi (2.5 - 5m)"
        elif w < 10: return "3. Ô tô tránh (5 - 10m)"
        return "4. Mặt tiền (> 10m)"
        
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


def chart_histogram_shape_ratio(df):
    """Biểu đồ tỷ lệ hình dáng đất"""
    if 'front_width' not in df.columns: return None
    
    work_df = df[(df['front_width'] > 0) & (df['area'] > 0)].copy()
    work_df['shape_ratio'] = (work_df['area'] / work_df['front_width']) / work_df['front_width']
    work_df = work_df[work_df['shape_ratio'] <= 20]
    
    fig = px.histogram(
        work_df, x='shape_ratio', nbins=40,
        color_discrete_sequence=['#26A69A'],
        labels={'shape_ratio': 'Tỷ lệ (Chiều Dài / Chiều Rộng)'}
    )
    
    # --- [CẬP NHẬT 1: Đổi tên trục dọc (Trục Y) dứt điểm] ---
    fig.update_yaxes(title_text="Số lượng tin")
    
    # --- [CẬP NHẬT 2: Ép Tooltip (Hover) hiển thị tiếng Việt sạch sẽ] ---
    # %{x} là giá trị trục ngang (Tỷ lệ), %{y} là giá trị trục dọc (Số lượng)
    fig.update_traces(hovertemplate='Tỷ lệ (Dài/Rộng): %{x:.1f}<br>Số lượng tin: %{y}')
    
    fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Vuông (1:1)")
    fig.add_vline(x=4, line_dash="dot", line_color="orange", annotation_text="Nhà ống (4:1)")
    fig.update_layout(bargap=0.1)
    
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

# ==============================================================================
# 4. GIAO DIỆN CHÍNH (MAIN UI)
# ==============================================================================

def show_dashboard_ui(df, category_name, city_mode="All"):
    """
    Hàm hiển thị chính được gọi từ app.py
    """
    if df is None or df.empty:
        st.warning(f"⚠️ Chưa có dữ liệu cho danh mục: **{category_name}**")
        return

    # 1. KPI
    render_kpi_metrics(df)
    st.markdown("---")

    # 2. BẢN ĐỒ LỚN (Có Scroll Zoom)
    # [FIX MỚI] Dùng Streamlit để vẽ Tiêu đề Icon
    st.subheader(f":material/location_on: Bản đồ phân bố ({len(df)} tin)")
    st.plotly_chart(chart_heatmap_location(df, city_mode=city_mode), width="stretch", config={'scrollZoom': True})
    st.markdown("---")

    # 3. TOP DỰ ÁN (Hàm này đã có sẵn st.subheader bên trong nên không cần thêm)
    chart_top_expensive_projects(df, city_mode=city_mode)
    st.markdown("---")

    # 4. PHÂN TÍCH SÂU
    c1, c2 = st.columns(2)
    with c1:
        # [FIX MỚI] Tiêu đề biểu đồ Tròn
        st.subheader(f":material/balance: Cơ cấu Pháp lý")
        st.plotly_chart(chart_donut_legal(df), width="stretch")
    with c2:
        # [FIX MỚI] Tiêu đề biểu đồ Xu hướng
        st.subheader(":material/trending_up: Xu hướng Diện tích - Giá")
        st.plotly_chart(chart_scatter_area_price(df), width="stretch")

    # 5. BIỂU ĐỒ ĐẶC THÙ
    if 'access_road' in df.columns:
        st.markdown("---")
        # [FIX MỚI] Tiêu đề biểu đồ Boxplot
        st.subheader(":material/add_road: Phân phối giá theo loại đường")
        st.plotly_chart(chart_box_alley_impact(df), width="stretch")
        
    if 'front_width' in df.columns:
        # [FIX MỚI] Tiêu đề biểu đồ Histogram
        st.subheader(":material/architecture: Phân phối Hình dáng đất (Dài/Rộng)")
        st.plotly_chart(chart_histogram_shape_ratio(df), width="stretch")