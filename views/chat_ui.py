import streamlit as st
import time
from src import chatbot_engine

def show_chat_interface():
    """
    Hàm hiển thị giao diện Chatbot hoàn chỉnh.
    """
    
    # ==============================================================================
    # 1. HEADER GIAO DIỆN
    # ==============================================================================
    st.markdown("## :material/smart_toy: Trợ lý ảo AI Bất động sản")
    st.caption(":material/rocket_launch: Hỗ trợ giải đáp pháp lý, quy trình mua bán và kiến thức nhà đất (Powered by Gemini 1.5 Flash)")
    
    # Nút xóa lịch sử chat (Để reset khi cần)
    if st.button(":material/delete_outline: Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun() # Chạy lại app để làm mới giao diện ngay lập tức

    # ==============================================================================
    # 2. KHỞI TẠO BỘ NHỚ (SESSION STATE)
    # ==============================================================================
    # Nếu chưa có biến 'messages' trong túi thần kỳ, tạo mới nó.
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        # Thêm câu chào mừng mặc định từ AI
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Chào bạn! Tôi là Trợ lý ảo Bất động sản. Tôi có thể giúp gì cho bạn về thủ tục pháp lý, quy hoạch hay thị trường hôm nay?"
        })

    # ==============================================================================
    # 3. HIỂN THỊ LỊCH SỬ CHAT (RENDER UI)
    # ==============================================================================
    # Vòng lặp này vẽ lại toàn bộ tin nhắn cũ mỗi khi Streamlit chạy lại (Rerun).
    for message in st.session_state.messages:
        # [ĐÃ XÓA] dòng if else chọn avatar Emoji cứng
        
        # Chỉ truyền đúng role, Streamlit sẽ tự bung Icon xịn kèm định danh cho CSS
        with st.chat_message(message["role"]): 
            st.markdown(message["content"])

    # ==============================================================================
    # 4. XỬ LÝ NHẬP LIỆU (USER INPUT)
    # ==============================================================================
    # Hàm st.chat_input sẽ tạo ô nhập liệu dính ở dưới cùng màn hình.
    if prompt := st.chat_input("Nhập câu hỏi của bạn (VD: Thủ tục sang tên sổ đỏ?)..."):
        
        # A. Hiển thị câu hỏi của người dùng ngay lập tức
        with st.chat_message("user"): # [ĐÃ XÓA] tham số avatar="👤"
            st.markdown(prompt)
        
        # B. Lưu câu hỏi vào bộ nhớ (để lần sau Rerun không bị mất)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # C. Gọi AI trả lời (Có hiệu ứng Loading)
        with st.chat_message("assistant"): # [ĐÃ XÓA] tham số avatar="🤖"
            
            # Tạo hiệu ứng "Đang suy nghĩ..." để người dùng biết app đang chạy
            with st.spinner("AI đang phân tích câu hỏi..."):
                
                # Gọi hàm xử lý từ file chatbot_engine.py
                ai_reply = chatbot_engine.get_gemini_response(
                    prompt, 
                    st.session_state.messages[:-1] # Trừ câu vừa nhập ra để tránh trùng lặp
                )
                
                # Hiển thị câu trả lời ra màn hình
                st.markdown(ai_reply)
        
        # D. Lưu câu trả lời của AI vào bộ nhớ
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})