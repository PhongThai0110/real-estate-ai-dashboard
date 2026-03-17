import streamlit as st
import time
from src import chatbot_engine

# Cấu hình Avatar
BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712139.png" 
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png" 

def stream_generator(text, delay=0.015):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)

def show_chat_interface():
    # ==============================================================================
    # 1. HEADER GIAO DIỆN
    # ==============================================================================
    st.markdown("## :material/support_agent: Trợ lý Cố vấn Bất động sản")
    st.caption(":material/hub: Hỗ trợ giải đáp pháp lý, phân tích dữ liệu thị trường và tư vấn chiến lược (Powered by Multi-Agent AI: Llama-3, Qwen & Gemini)")
    
    # Nút xóa lịch sử (Đã được CSS "làm mềm" lại thành dạng Outline viền xám)
    if st.button(":material/delete_outline: Xóa lịch sử chat", type="secondary"):
        st.session_state.messages = []
        st.rerun() 

    # ==============================================================================
    # 2. KHỞI TẠO BỘ NHỚ
    # ==============================================================================
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Chào bạn! Tôi là Trợ lý Cố vấn Bất động sản. Tôi có thể giúp gì cho bạn về thủ tục pháp lý, quy hoạch hay phân tích thị trường hôm nay?"
        })
        
    if "suggested_prompt" not in st.session_state:
        st.session_state.suggested_prompt = None

    # ==============================================================================
    # 3. HIỂN THỊ NÚT GỢI Ý
    # ==============================================================================
    if len(st.session_state.messages) == 1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 📢 SỬA LỖI 1: KHẮC PHỤC LỖI MARKDOWN BẰNG HTML THUẦN
        # Dùng thẻ <p> thay vì ** để chữ in đậm, đẹp và không bao giờ lộ code
        st.markdown("<p style='font-weight: 500; font-size: 14.5px; color: #999999; margin-bottom: 15px;'> Gợi ý câu hỏi để thử nghiệm hệ thống:</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Tìm nhà Cầu Giấy 15 Tỷ",type="tertiary", use_container_width=True):
                st.session_state.suggested_prompt = "Tôi có 15 tỷ. Tôi đang phân vân giữa việc mua nhà lô góc ở Cầu Giấy và một lô đất nền ở Hoài Đức. Hãy tìm cho tôi 1 căn nhà phố Cầu Giấy đắt nhất (trong tầm giá 15 tỷ) và 1 lô đất nền Hoài Đức diện tích to nhất (cũng trong tầm giá 15 tỷ). In rõ thông tin từng căn ra."
        with col2:
            if st.button("Hỏi Luật Đầu cơ & Khống chế giá",type="tertiary", use_container_width=True):
                st.session_state.suggested_prompt = "Tôi thấy dạo này giá nhà chung cư đang tăng ảo. Theo các văn bản pháp lý hiện hành từ năm 2024 trở đi, có quy định nào khống chế giá trần chung cư hay chống đầu cơ thổi giá không?"
        with col3:
            if st.button("Tính Thuế Phí", type="tertiary", use_container_width=True):
                st.session_state.suggested_prompt = "Tìm cho tôi căn biệt thự VIP có giá rẻ nhất ở Quận 3. Dựa trên mức giá của căn đó, hãy tính tổng chi tiết các loại thuế, phí trước bạ và phí công chứng mà tôi nộp theo luật 2026. Tổng chi phí cuối cùng là bao nhiêu?"
        st.markdown("---")

    # ==============================================================================
    # 4. HIỂN THỊ LỊCH SỬ CHAT
    # ==============================================================================
    for message in st.session_state.messages:
        avatar_img = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar_img): 
            st.markdown(message["content"])

    # ==============================================================================
    # 5. XỬ LÝ NHẬP LIỆU
    # ==============================================================================
    prompt = st.chat_input("Nhập câu hỏi của bạn (VD: Thủ tục sang tên sổ đỏ?)...") or st.session_state.suggested_prompt

    if prompt:
        st.session_state.suggested_prompt = None
        
        with st.chat_message("user", avatar=USER_AVATAR): 
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar=BOT_AVATAR): 
            with st.spinner("AI đang tra cứu dữ liệu và pháp lý..."):
                ai_reply = chatbot_engine.chat_router(
                    prompt, 
                    st.session_state.messages[:-1] 
                )
            
            st.write_stream(stream_generator(ai_reply))
        
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})