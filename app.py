import streamlit as st
import pandas as pd
import sys
import os
import numpy as np

from views import sidebar
from views import dashboard
from src import loader
from src import preprocessor
from src import explainer
from views import chat_ui

def local_css(file_name):
    try:
        with open(file_name,encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass 

st.set_page_config(
    page_title="Real Estate AI",
    page_icon=":material/domain:",
    layout="wide",
    initial_sidebar_state="expanded"
)
local_css("assets/style.css")

def execute_prediction_flow(user_inputs, city_mode, property_type):
    if property_type == "Nhà phố":
        process_key = f"Nhà phố {city_mode}"  
    elif property_type == "Căn hộ Chung cư": 
        process_key = "Căn hộ Chung cư"
    elif property_type == "Đất nền":
        process_key = "Đất nền"
    else:
        process_key = "Biệt thự / Villa"

    system_resources = loader.load_models(city_mode, property_type)
    
    if not system_resources or 'model' not in system_resources:
        st.error("❌ Không tìm thấy Model. Hãy kiểm tra folder models/.")
        return None

    model = system_resources['model']

    try:
        processed_df = preprocessor.transform_input(user_inputs, process_key)
    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
        return None
    
    try:
        if hasattr(model, 'feature_names_in_'):
            required_cols = list(model.feature_names_in_)
            current_cols = list(processed_df.columns)
            
            missing_cols = []
            for col in required_cols:
                if col not in current_cols:
                    processed_df[col] = 0
                    missing_cols.append(col)
            
            if missing_cols:
                print("\n" + "="*40)
                print(f"⚠️ CẢNH BÁO: Model {process_key} thiếu các cột sau (đã fill 0):")
                print(missing_cols)
                
                if property_type == "Căn hộ Chung cư":
                    pj_val = processed_df.get('project_name', pd.Series([0])).iloc[0]
                    floor_val = processed_df.get('floors', pd.Series([0])).iloc[0]
                    print(f"🧐 Project Name Value: {pj_val}")
                    print(f"🧐 Floors Value: {floor_val}")
                print("="*40 + "\n")

            processed_df = processed_df[required_cols]
    except Exception as e:
        print(f"Lỗi khớp cột: {e}")
        pass

    try:
        pred_log = model.predict(processed_df)[0]
        
        pred_real = np.expm1(pred_log) 
    
        return max(0, pred_real), model, processed_df  
    except Exception as e:
        st.error(f"Lỗi khi model dự báo: {e}")
        return None

def format_currency(amount):
    if amount >= 1: return f"{amount:,.2f} Tỷ"
    return f"{amount*1000:,.0f} Triệu"

def main():
    nav_mode, user_inputs, third_param, property_type, submit_btn = sidebar.show_sidebar()
    
    if nav_mode == ":material/dashboard: Dashboard Phân tích":
        dashboard_category = third_param  
        st.title(f":material/monitoring: Phân tích: {dashboard_category}")
        
        all_data = loader.load_raw_data()
        
        map_key = {
            "Nhà phố Hồ Chí Minh": "hcm",
            "Nhà phố Hà Nội": "hanoi",
            "Căn hộ Chung cư": "apartment",
            "Đất nền": "land",
            "Biệt thự / Villa": "villa"
        }
        
        selected_key = map_key.get(dashboard_category)
        df_selected = all_data.get(selected_key)
        if "Hồ Chí Minh" in dashboard_category:
            filter_city_mode = "Hồ Chí Minh"
        elif "Hà Nội" in dashboard_category:
            filter_city_mode = "Hà Nội"
        else:
            filter_city_mode = "All"

        if df_selected is not None and not df_selected.empty:
            dashboard.show_dashboard_ui(
                df_selected, 
                dashboard_category, 
                city_mode=filter_city_mode 
            )
        else:
            st.warning(f"⚠️ Không tìm thấy dữ liệu cho **{dashboard_category}**.")
            
    elif nav_mode == ":material/chat: Chatbot Tư vấn":
        chat_ui.show_chat_interface()
        
    else:
        city_mode = third_param 
        
        st.title(":material/psychology: AI Định giá Bất động sản")
        
        if not submit_btn:
            st.info(":material/edit_document: Vui lòng nhập thông tin BĐS bên thanh Sidebar để bắt đầu định giá.")
            if os.path.exists("assets/banner_intro.png"):
                st.image("assets/banner_intro.png", width="stretch")
        else:
            if user_inputs['area'] <= 0:
                st.error("⚠️ Diện tích phải lớn hơn 0 m².")
            else:
                with st.spinner("AI đang phân tích và định giá..."):
                    price, model, processed_data = execute_prediction_flow(user_inputs, city_mode, property_type)
                
                if price:
                    st.success("✅ Dự báo thành công!")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Giá tham khảo", format_currency(price))
                    with c2:
                        don_gia = (price * 1000) / user_inputs['area']
                        st.metric("Đơn giá ước tính", f"{don_gia:,.1f} Tr/m²")
                    
                    st.caption("*Kết quả chỉ mang tính chất tham khảo.*")
                    
                    st.markdown("---")
                    st.subheader(":material/insights: Tại sao AI đưa ra mức giá này?")
                    
                    with st.expander("xem chi tiết phân tích", expanded=True):
                        if model and processed_data is not None:
                            df_expl, base_val = explainer.get_explanation(model, processed_data)
                            
                            if df_expl is not None:
                                fig = explainer.plot_waterfall(df_expl)
                                st.subheader(":material/waterfall_chart: Mức độ ảnh hưởng đến giá nhà")
                                st.plotly_chart(fig, use_container_width=True)
                                
                            top_pos = df_expl[df_expl['Contribution'] > 0].head(1)
                            top_neg = df_expl[df_expl['Contribution'] < 0].head(1)
                            
                            st.write("**Phân tích nhanh:**")
                            
                            if not top_pos.empty:
                                feat_name = top_pos.iloc[0]['Feature']
                                st.markdown(
                                    f"- <span style='color:#00C853; font-size:1.2em;'>:material/trending_up:</span> **Yếu tố làm TĂNG giá mạnh nhất:** {feat_name}", 
                                    unsafe_allow_html=True
                                )

                            if not top_neg.empty:
                                feat_name = top_neg.iloc[0]['Feature']
                                st.markdown(
                                    f"- <span style='color:#FF5252; font-size:1.2em;'>:material/trending_down:</span> **Yếu tố làm GIẢM giá mạnh nhất:** {feat_name}", 
                                    unsafe_allow_html=True
                                )

if __name__ == "__main__":
    main()