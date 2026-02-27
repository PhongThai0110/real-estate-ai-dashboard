import google.generativeai as genai
import streamlit as st
import os

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (SYSTEM CONFIGURATION)
# ==============================================================================

def configure_api():
    """
    Hàm cấu hình API Key an toàn.
    Lấy key từ file .streamlit/secrets.toml để tránh lộ key trong code.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return True
    except KeyError:
        st.error("⚠️ Lỗi: Chưa tìm thấy 'GEMINI_API_KEY' trong file .streamlit/secrets.toml")
        return False
    except Exception as e:
        st.error(f"⚠️ Lỗi kết nối Google AI: {str(e)}")
        return False

# ==============================================================================
# 2. THIẾT KẾ SYSTEM PROMPT (LUẬT LỆ CỐT LÕI)
# ==============================================================================
# Đây là phần quan trọng nhất để Bot thông minh và không trả lời linh tinh.

SYSTEM_INSTRUCTION = """
VAI TRÒ:
Bạn là 'Real Estate AI' - Một trợ lý ảo chuyên nghiệp, am hiểu sâu sắc về thị trường Bất động sản Việt Nam. 
Bạn được phát triển để hỗ trợ người dùng định giá, tìm hiểu pháp lý và quy trình giao dịch nhà đất.

NHIỆM VỤ CHÍNH:
1. Giải thích các khái niệm pháp lý (Sổ đỏ, Sổ hồng, Vi bằng, Quy hoạch 1/500...).
2. Tư vấn quy trình mua bán, sang tên, công chứng.
3. Phân tích các yếu tố ảnh hưởng đến giá nhà (Vị trí, hẻm, phong thủy cơ bản).
4. Đưa ra lời khuyên an toàn khi giao dịch để tránh lừa đảo.

QUY TẮC ỨNG XỬ (GUARDRAILS) - BẮT BUỘC TUÂN THỦ:
1. PHẠM VI: CHỈ trả lời câu hỏi liên quan đến Bất động sản, Nhà cửa, Kiến trúc, Tài chính mua nhà.
2. TỪ CHỐI: Nếu người dùng hỏi chủ đề khác (Code, Toán học, Chính trị, Giải trí, Nấu ăn...), hãy từ chối lịch sự.
   - Mẫu câu từ chối: "Xin lỗi, tôi là trợ lý chuyên về Bất động sản, tôi không có dữ liệu để trả lời câu hỏi này."
3. GIỌNG ĐIỆU: Chuyên nghiệp, khách quan, hữu ích và thân thiện. Dùng Tiếng Việt chuẩn.
4. CẬP NHẬT PHÁP LÝ (QUAN TRỌNG):
   - Tuyệt đối KHÔNG nhắc đến "Sổ hộ khẩu" hoặc "Sổ tạm trú" (đã bãi bỏ). Thay vào đó hãy dùng "Căn cước công dân (CCCD) gắn chip" hoặc "Tài khoản định danh điện tử (VNeID)".
   - Luôn nhắc người dùng mang theo bản gốc giấy tờ khi đi công chứng.
5. ĐỊNH DẠNG: Sử dụng Markdown (in đậm **...**, gạch đầu dòng - ) để câu trả lời dễ đọc.
6. KHÔNG BỊA ĐẶT: Nếu không biết câu trả lời (ví dụ thông tin quy hoạch quá mới), hãy thành thật nói không biết và khuyên người dùng tra cứu tại cơ quan chức năng.
"""

# ==============================================================================
# 3. HÀM XỬ LÝ CHAT (CORE LOGIC)
# ==============================================================================

def get_gemini_response(user_prompt, chat_history):
    """
    Gửi câu hỏi và lịch sử chat lên Google Gemini để nhận phản hồi.
    
    Args:
        user_prompt (str): Câu hỏi hiện tại của người dùng.
        chat_history (list): Danh sách các tin nhắn cũ từ Session State.
        
    Returns:
        str: Câu trả lời từ AI hoặc thông báo lỗi.
    """
    
    # 1. Kiểm tra cấu hình API
    if not configure_api():
        return "Hệ thống chưa được cấu hình API Key. Vui lòng kiểm tra lại."

    try:
        # 2. Khởi tạo Model với Config tối ưu
        # generation_config giúp kiểm soát độ sáng tạo. 
        # temperature=0.7 là mức cân bằng giữa sáng tạo và chính xác.
        model = genai.GenerativeModel(
            model_name="gemini-flash-latest",
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config={"temperature": 0.7, "max_output_tokens": 4096}
        )

        # 3. Chuyển đổi lịch sử chat (Streamlit format -> Gemini format)
        # Streamlit dùng: {"role": "user"/"assistant", "content": "..."}
        # Gemini dùng:    {"role": "user"/"model", "parts": ["..."]}
        gemini_history = []
        for msg in chat_history:
            # Bỏ qua tin nhắn hệ thống hoặc lỗi nếu có
            if msg["role"] not in ["user", "assistant"]:
                continue
                
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })

        # 4. Bắt đầu phiên chat và gửi tin nhắn
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(user_prompt)
        
        return response.text

    except Exception as e:
        # Xử lý lỗi (ví dụ: Mất mạng, API hết hạn mức...)
        return f"Xin lỗi, tôi đang gặp sự cố kết nối. Chi tiết lỗi: {str(e)}"