import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# TỪ ĐIỂN VIỆT HÓA (Mapping)
FEATURE_MAP = {
    "area": "Diện tích đất",
    "front_width": "Mặt tiền ngang",
    "access_road": "Độ rộng đường/hẻm",
    "bedrooms": "Số phòng ngủ",
    "floors": "Số tầng",
    "toilet": "Số Toilet",
    "legal": "Pháp lý",
    "direction": "Hướng nhà",
    "interior": "Nội thất",
    "project_name": "Dự án",
    "is_corner": "Lô góc (2 mặt tiền)",
    "interior_score": "Điểm nội thất",
    "legal_score": "Điểm pháp lý"

}

def get_explanation(model, X_input):
    """
    Tính toán SHAP và GỘP BIẾN Lat/Lon thành 'Vị trí'.
    """
    try:
        # 1. Tính SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_input)
        
        values = shap_values.values[0]
        feature_names = list(X_input.columns)
        
        # 2. Tạo DataFrame ban đầu
        df_expl = pd.DataFrame({
            'Feature_Raw': feature_names,
            'Contribution': values
        })
        
        # ==========================================================
        # [LOGIC MỚI] GỘP LAT & LON THÀNH 'VỊ TRÍ & KHU VỰC'
        # ==========================================================
        
        # Lọc lấy giá trị của lat và lon
        lat_val = df_expl.loc[df_expl['Feature_Raw'] == 'lat', 'Contribution'].sum()
        lon_val = df_expl.loc[df_expl['Feature_Raw'] == 'lon', 'Contribution'].sum()
        
        # Tổng hợp tác động của vị trí
        location_impact = lat_val + lon_val
        
        # Loại bỏ dòng lat và lon cũ
        df_expl = df_expl[~df_expl['Feature_Raw'].isin(['lat', 'lon'])]
        
        # Thêm dòng mới 'Vị trí'
        new_row = pd.DataFrame([{
            'Feature_Raw': 'LOCATION_GROUP', 
            'Contribution': location_impact
        }])
        df_expl = pd.concat([df_expl, new_row], ignore_index=True)
        
        # ==========================================================
        # [LOGIC MỚI] VIỆT HÓA TÊN GỌI
        # ==========================================================
        def map_name(raw_name):
            if raw_name == 'LOCATION_GROUP': return "📍 Vị trí & Khu vực"
            return FEATURE_MAP.get(raw_name, raw_name) # Nếu không có trong từ điển thì giữ nguyên
            
        df_expl['Feature'] = df_expl['Feature_Raw'].apply(map_name)
        
        # 3. Sắp xếp lại theo độ ảnh hưởng tuyệt đối
        df_expl['Abs_Contribution'] = df_expl['Contribution'].abs()
        df_expl = df_expl.sort_values(by='Abs_Contribution', ascending=False)
        
        base_value = shap_values.base_values[0]
        return df_expl, base_value

    except Exception as e:
        st.error(f"Lỗi SHAP Explainer: {e}")
        return None, None

def plot_waterfall(df_expl):
    """
    Vẽ biểu đồ Waterfall với màu sắc trực quan.
    """
    # Lấy Top 7 yếu tố quan trọng nhất
    top_features = df_expl.head(7).iloc[::-1]
    
    # Định dạng màu sắc: Xanh lá (Tăng giá) / Đỏ (Giảm giá)
    colors = ['#00C853' if x > 0 else '#FF5252' for x in top_features['Contribution']]
    
    # Định dạng text hiển thị (Thêm dấu + nếu dương)
    text_labels = [f"{x:+.2f} (Log)" for x in top_features['Contribution']]
    
    fig = go.Figure(go.Bar(
        x=top_features['Contribution'],
        y=top_features['Feature'],
        orientation='h',
        marker_color=colors,
        text=text_labels,
        textposition='auto',
        hoverinfo='y+x'
    ))
    
    fig.update_layout(
        title={
            'text': "🔍 Mức độ ảnh hưởng đến giá nhà",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color='#38bdf8')
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=13),
        xaxis=dict(
            title="Mức độ tác động (Sang phải là Tăng giá / Sang trái là Giảm giá)", 
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='white'
        ),
        yaxis=dict(title=""),
        margin=dict(l=10, r=10, t=100, b=10), 
        height=450
    )
    
    return fig