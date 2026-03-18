import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# TỪ ĐIỂN VIỆT HÓA (Mapping cơ bản)
FEATURE_MAP = {
    "area": "Diện tích đất",
    "front_width": "Mặt tiền ngang",
    "access_road": "Độ rộng đường/hẻm",
    "bedrooms": "Số phòng ngủ",
    "bathrooms": "Số phòng tắm",
    "floors": "Số tầng",
    "toilet": "Số Toilet",
    "road_class": "Phân loại đường",
    "interior": "Nội thất",
    "project_name": "Dự án",
    "is_corner": "Lô góc (2 mặt tiền)",
    "interior_score": "Chất lượng nội thất",
    "legal_score": "Điểm pháp lý",
    "log_area": "Quy mô Diện tích (Log)", 
    "land_depth": "Chiều sâu thửa đất (Giả định)",
    "shape_ratio": "Tỷ lệ Hình dáng (Dài/Rộng)",
    "business_potential": "Tiềm năng Kinh doanh (Mặt tiền x Đường)",
    "log_dist": "Khoảng cách đến Trung tâm (Log)",
    "geo_cluster": "Cụm Vị trí / Khu vực",
    "interior_encoded": "Chất lượng nội thất",
}

def get_explanation(model, X_input):
    """
    Tính toán SHAP và GỘP BIẾN (Location, Direction, Legal) theo chuẩn chuyên gia.
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
        # [LOGIC CHUYÊN GIA] GỘP CÁC BIẾN CÙNG NHÓM (SHAP ADDITIVITY)
        # ==========================================================
        
        # --- A. GỘP TỌA ĐỘ THÀNH 'VỊ TRÍ' ---
        if 'lat' in feature_names and 'lon' in feature_names:
            loc_impact = df_expl.loc[df_expl['Feature_Raw'].isin(['lat', 'lon']), 'Contribution'].sum()
            df_expl = df_expl[~df_expl['Feature_Raw'].isin(['lat', 'lon'])]
            df_expl = pd.concat([df_expl, pd.DataFrame([{'Feature_Raw': 'LOCATION_GROUP', 'Contribution': loc_impact}])], ignore_index=True)
            
        # --- B. GỘP ONE-HOT HƯỚNG NHÀ (DIRECTION) ---
        dir_cols = [col for col in feature_names if col.startswith('direction_')]
        if dir_cols:
            # 1. Cộng tổng lực tác động của TẤT CẢ các cột hướng
            dir_impact = df_expl.loc[df_expl['Feature_Raw'].isin(dir_cols), 'Contribution'].sum()
            
            # 2. Tìm xem thực tế User đã nhập hướng nào (Cột nào = 1)
            selected_dir = "Chưa xác định"
            for col in dir_cols:
                if X_input[col].iloc[0] == 1:
                    selected_dir = col.replace('direction_', '')
                    break
                    
            # 3. Xóa các cột lẻ, tạo 1 dòng gộp duy nhất có tên linh hoạt
            df_expl = df_expl[~df_expl['Feature_Raw'].isin(dir_cols)]
            df_expl = pd.concat([df_expl, pd.DataFrame([{'Feature_Raw': f'DIRECTION_GROUP_{selected_dir}', 'Contribution': dir_impact}])], ignore_index=True)

        # --- C. GỘP ONE-HOT PHÁP LÝ (LEGAL) ---
        legal_cols = [col for col in feature_names if col.startswith('legal_') and col != 'legal_score']
        if legal_cols:
            legal_impact = df_expl.loc[df_expl['Feature_Raw'].isin(legal_cols), 'Contribution'].sum()
            
            selected_legal = "Chưa xác định"
            for col in legal_cols:
                if X_input[col].iloc[0] == 1:
                    selected_legal = col.replace('legal_', '')
                    break
                    
            df_expl = df_expl[~df_expl['Feature_Raw'].isin(legal_cols)]
            df_expl = pd.concat([df_expl, pd.DataFrame([{'Feature_Raw': f'LEGAL_GROUP_{selected_legal}', 'Contribution': legal_impact}])], ignore_index=True)

        # ==========================================================
        # VIỆT HÓA TÊN GỌI CUỐI CÙNG
        # ==========================================================
        def map_name(raw_name):
            if raw_name == 'LOCATION_GROUP': 
                return "Vị trí & Khu vực"
            if str(raw_name).startswith('DIRECTION_GROUP_'): 
                val = raw_name.replace('DIRECTION_GROUP_', '')
                return f"Hướng: {val}"
            if str(raw_name).startswith('LEGAL_GROUP_'): 
                val = raw_name.replace('LEGAL_GROUP_', '')
                return f"Pháp lý: {val}"
            return FEATURE_MAP.get(raw_name, raw_name)
            
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
    Vẽ biểu đồ Waterfall với màu sắc trực quan (Đã chuyển Log thành %).
    """
    top_features = df_expl.head(7).iloc[::-1].copy()
    
    top_features['Impact_Percent'] = (np.exp(top_features['Contribution']) - 1) * 100
    
    colors = ['#00C853' if x > 0 else '#FF5252' for x in top_features['Impact_Percent']]
    text_labels = [f"{x:+.1f}%" for x in top_features['Impact_Percent']]
    
    fig = go.Figure(go.Bar(
        x=top_features['Impact_Percent'], 
        y=top_features['Feature'],
        orientation='h',
        marker_color=colors,
        text=text_labels,
        textposition='auto',
        hoverinfo='y+x'
    ))
    
    fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=13),
            xaxis=dict(
                title="Mức độ tác động (% Tăng/Giảm)", 
                showgrid=True, 
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=True,
                zerolinecolor='white'
            ),
            yaxis=dict(title=""),
            margin=dict(l=10, r=10, t=20, b=65), 
            height=480,
            annotations=[
                dict(
                    x=0.5,           
                    y=-0.21,         
                    xref='paper',
                    yref='paper',
                    text="<i>*Ghi chú: Mức độ tác động được so sánh với mặt bằng giá trung bình của toàn khu vực.</i>",
                    showarrow=False,
                    font=dict(size=12, color='#94a3b8') 
                )
            ]
        )
    return fig