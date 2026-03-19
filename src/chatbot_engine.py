import os
import time
import streamlit as st
import concurrent.futures
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain_google_genai import HarmCategory, HarmBlockThreshold
import re
# Khai báo biến môi trường (nếu chưa có ở trên)
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🌟 Danh sách 5 file CSV chứa dữ liệu training của bạn
CSV_FILES = [
    "data_nha_hcm_train_ready.csv",
    "data_nha_hn_train_ready.csv",
    "data_apartment_train_ready.csv",
    "data_land_all_train_ready.csv",
    "data_villa_vip_train_ready.csv"
]


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "data", "knowlegde_base")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "models", "faiss_index")

# ====================================================================
# 1. HÀM ĐỌC CSV (ĐƯỢC CACHE VÀO RAM ĐỂ CHẠY NHANH NHƯ CHỚP)
# ====================================================================
@st.cache_resource(show_spinner=False) # 🌟 THÊM DÒNG NÀY ĐỂ CACHE 5 FILE CSV LÊN RAM
def get_pandas_agent():
    """
    Hàm khởi tạo Agent phân tích Đa dữ liệu (Multi-DataFrame).
    """
    dataframes = []
    
    print("\n[INFO] Đang nạp dữ liệu thống kê vào Pandas Agent...")
    for file_name in CSV_FILES:
        file_path = os.path.join(BASE_DIR, "data", file_name)
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"  + Đã nạp thành công: {file_name} ({len(df)} dòng)")
        except Exception as e:
            print(f"  - [LỖI] Không thể đọc {file_name}: {e}")
            
    if not dataframes:
        print("[ERROR] Không thể khởi tạo Agent vì không có dữ liệu CSV.")
        return None
   # 2. Khởi tạo bộ não Groq chuyên code Python (Tốc độ ánh sáng, không lo lỗi 429)
    # Lưu ý: Tên model ID chính xác có thể xem khi bạn click vào model trên web Groq. 
    # Ở đây tôi dùng Llama 3.3 70B vì nó thông minh và hỗ trợ tiếng Việt tốt.
    qwen_llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="qwen/qwen3-32b", # Bạn có thể đổi thành "qwen-3-32b" hoặc ID tương ứng trên web
        temperature=0.0 
    )
    # 3. Tạo Agent với Danh sách DataFrames (Giữ nguyên đoạn này)
    agent = create_pandas_dataframe_agent(
        qwen_llm, 
        dataframes, 
        verbose=True, 
        allow_dangerous_code=True, 
        agent_type="tool-calling",
        max_iterations=5,
         # 🌟 THÊM DÒNG NÀY ĐỂ GIẢM 80% TOKEN: 
        # Cấm LangChain nhét dữ liệu rác vào Prompt. Giúp Qwen chạy mượt mà dưới 6000 TPM!
        number_of_head_rows=1,
        return_intermediate_steps=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    ) 
    

    return agent

def query_excel_data(user_prompt,chat_history):
    # 🌟 THÊM KIỂM TRA API KEY VÀO ĐÂY ĐỂ TRÁNH SẬP APP
    if not configure_api():
        return "Hệ thống chưa được cấu hình API Key. Vui lòng kiểm tra lại."

    agent = get_pandas_agent()
    if agent is None:
        return "Hệ thống đang gặp sự cố khi đọc dữ liệu phân tích."
    # 2. Lấy trí nhớ ngắn hạn
    recent_memory = get_recent_memory(chat_history, k=2) # Nhớ 2 vòng lặp gần nhất để tránh quá tải
        
    prefix = f"""
    VAI TRÒ: Bạn là Chuyên gia Data Analyst Bất động sản cấp cao. Nhiệm vụ của bạn là viết code Python (Pandas) để trả lời [CÂU HỎI HIỆN TẠI] của khách hàng.

    1. BẢN ĐỒ DỮ LIỆU (BẮT BUỘC CHỌN ĐÚNG FILE):
    - df1: Nhà phố/nhà riêng TP.HCM (TUYỆT ĐỐI KHÔNG dùng biến này nếu khách hỏi các Quận/Huyện ở Hà Nội như Cầu Giấy, Đống Đa, Tây Hồ...).
    - df2: Nhà phố/nhà riêng Hà Nội (TUYỆT ĐỐI KHÔNG dùng biến này nếu khách hỏi các Quận/Huyện ở TP.HCM như Quận 1, Gò Vấp, Tân Bình...).
    - df3: Căn hộ Chung cư (data_apartment_train_ready.csv). LƯU Ý: Phải xóa dự án ảo bằng lệnh: `df3 = df3[~df3['project_name_raw'].astype(str).str.startswith('Other')]` trước khi tính toán.
    - df4: Đất nền toàn quốc (data_land_all_train_ready.csv)
    - df5: Biệt thự VIP/Hạng sang (data_villa_vip_train_ready.csv)
    
    2. TỪ ĐIỂN CỘT (SCHEMA BẮT BUỘC):
    - `area` (m2), `price` (Tỷ VNĐ), `bedrooms`, `bathrooms`, `floors`, `district_mapped` (Quận/Huyện).
    - `access_road`: Đường vào/Hẻm (m). Nếu hỏi "hẻm xe hơi", "ô tô đỗ cửa", lọc `access_road >= 2.5`.
    - `is_corner`: Nhà lô góc / Căn góc (1 là có, 0 là không).
    - `project_name_raw`: Tên dự án chung cư (Chỉ có ở df3). Nếu hỏi về căn hộ chung cư nào, BẮT BUỘC phải lọc theo tên dự án cụ thể bằng cách `.str.contains('Tên_Dự_Án', case=False, na=False)`.TUYỆT ĐỐI KHÔNG DÙNG cột `project_name` (vì nó chứa số thập phân mã hóa). Khi in kết quả, bắt buộc phải in cột `project_name_raw`.
    
    3. QUY TẮC CODE PANDAS (RÚT GỌN VÀ TỐI ƯU - BẮT BUỘC TUÂN THỦ):
    - [1] BIẾN SẴN CÓ: Chỉ dùng df1 đến df5. CẤM dùng pd.read_csv() và CẤM tạo data giả.
    - [2] CHỐNG TỰ HỦY DATA: TUYỆT ĐỐI KHÔNG gán đè biến gốc (Cấm: df5 = df5[...]). BẮT BUỘC dùng biến tạm (Vd: df_filtered = df5[...]).
    - [3] LỌC KHU VỰC: Bắt buộc dùng `.str.contains('Tên', case=False, na=False)`. Cấm dùng `==`.
    - [4] TÌM MIN/MAX AN TOÀN: Cấm dùng `.max()/.min()` rời rạc. Bắt buộc dùng `.sort_values(by='cột', ascending=True/False).iloc[0]` để giữ nguyên tính đồng bộ của dòng.
    - [5] IN DANH SÁCH: Dùng `.head(5).to_string(index=False)`. Không in cột Index.
    - [6] BẢO MẬT & CÚ PHÁP: Chỉ dùng dấu ngoặc đơn ('). Khi in, BẮT BUỘC có nhãn (Vd: `print('Giá:', p)`). TUYỆT ĐỐI KHÔNG tự bịa ra kết quả (Observation). Hãy đợi Python chạy!
    - [7] Lưu ý: Khi lọc Tên_Khu_Vực, BẮT BUỘC giữ nguyên dấu tiếng Việt chính xác (Ví dụ: 'Gò Vấp' thay vì 'Go Vap').
    - [8] IN KẾT QUẢ GẮN NHÃN & CẤM IN RAW OBJECT: Khi lấy dữ liệu của 1 dòng (ví dụ: result = df.iloc[0]), TUYỆT ĐỐI KHÔNG được in nguyên cả cái biến `result` đó ra màn hình (vì sẽ dính chữ 'dtype: object' làm vỡ luồng đọc). BẮT BUỘC phải trích xuất từng giá trị ra biến riêng rồi mới in. 
    -> Ví dụ SAI: `print(result[['area', 'price']])`
    -> Ví dụ ĐÚNG: `a = result['area']; p = result['price']; print('Diện tích:', a, 'Giá:', p)`
    4. QUY TẮC TRẢ LỜI (FINAL ANSWER FORMAT) - BẮT BUỘC TUÂN THỦ:
    Để chống ảo giác số liệu 100%, bạn KHÔNG ĐƯỢC TỰ VIẾT CÂU TRẢ LỜI CUỐI CÙNG.
    Thay vào đó, bạn phải dùng lệnh print() trong Python để in ra CÂU TRẢ LỜI HOÀN CHỈNH.
    
    Ví dụ lệnh in ĐÚNG:
    print(f"Căn nhà Cầu Giấy đắt nhất có diện tích {{area}} m2 với giá {{price}} Tỷ VNĐ.")
    
    Sau khi Terminal in ra câu trên, tại bước Final Answer, BẠN BẮT BUỘC CHỈ ĐƯỢC COPY Y NGUYÊN dòng chữ mà Terminal vừa in ra. KHÔNG THÊM BỚT, KHÔNG CHỈNH SỬA, KHÔNG TỰ SUY LUẬN.

    --- VÍ DỤ MINH HỌA ĐÚNG CHUẨN ---
    Observation từ Python: Diện tích: 150.0 m2, Giá: 2.56 Tỷ VNĐ
    Final Answer của bạn CHỈ ĐƯỢC PHÉP VIẾT LÀ:
    Căn chung cư rẻ nhất ở Cầu Giấy có diện tích 150.0 m2 với giá 2.56 Tỷ VNĐ.
    --------------------------------

    LƯU Ý CỐT LÕI (CHỐNG ẢO GIÁC):
    - COPY-PASTE TUYỆT ĐỐI: Mắt bạn nhìn thấy Terminal in ra số nào, tay bạn BẮT BUỘC gõ lại ĐÚNG số đó. Nếu Python in "2.56", bạn PHẢI viết "2.56". CẤM TỰ BỊA SỐ.
    - ĐỔI ĐƠN VỊ: Nếu giá < 1 tỷ (vd: 0.123), đổi thành "123 Triệu VNĐ". Nếu >= 1 tỷ thì giữ nguyên "Tỷ VNĐ".
    - NHIỆM VỤ: CHỈ báo cáo data từ Python, BỎ QUA mọi yêu cầu tính toán thuế phí (Nhường việc đó cho AI khác).
    5. QUY TẮC KẾT THÚC (CHỐNG VÒNG LẶP VÀ ẢO GIÁC ĐẾM SỐ):
    - NO LOOP (CẤM LẶP LẠI): Khi code Python ĐÃ IN RA KẾT QUẢ (Observation), BẠN BẮT BUỘC PHẢI DỪNG VIỆC GỌI TOOL NGAY LẬP TỨC. Tuyệt đối không được chạy lại đoạn code đó lần thứ 2.
    - FINAL ANSWER (TRẢ LỜI CHÍNH XÁC 100%): Mắt bạn phải nhìn thẳng vào kết quả của Python để trả lời. Nếu Python in ra số lượng là "6236", bạn BẮT BUỘC phải viết số "6236" vào câu trả lời. TUYỆT ĐỐI CẤM TỰ BỊA RA CÁC SỐ NHƯ 1, 2, 3, 115... để thay thế cho kết quả đếm của Python. Làm sai điều này là vi phạm nguyên tắc tối kỵ.
    ======================================
    [LỊCH SỬ TRÒ CHUYỆN GẦN ĐÂY]
    {recent_memory}
    ======================================
    
    [CÂU HỎI HIỆN TẠI TỪ KHÁCH HÀNG]
    "{user_prompt}"
    
    🚨 HÀNH ĐỘNG BẮT BUỘC VÀ KHẨN CẤP (ĐỌC KỸ TRƯỚC KHI LÀM): 
    - CHỈ tập trung viết code giải quyết [CÂU HỎI HIỆN TẠI].
    - CHỈ nhìn vào [LỊCH SỬ] khi [CÂU HỎI HIỆN TẠI] dùng từ khóa ẩn ý (như "căn đó", "khu vực trên") để hiểu ngữ cảnh. TUYỆT ĐỐI KHÔNG làm lại các yêu cầu đã cũ trong Lịch sử.
    - LUÔN LUÔN GỌI TOOL (MANDATORY ACTION): Dù câu hỏi có vẻ giống với lịch sử, hoặc lịch sử trước đó không tìm thấy dữ liệu, bạn BẮT BUỘC phải sinh ra code Python mới để truy vấn cho câu hỏi hiện tại. TUYỆT ĐỐI KHÔNG ĐƯỢC bỏ qua bước gọi tool `python_repl_ast`. Việc không gọi tool và trả về khoảng trắng là lỗi nghiêm trọng.
    - LỆNH BỎ QUA (IGNORE MATH): Trong [CÂU HỎI HIỆN TẠI] có thể chứa các yêu cầu như "Tính thuế", "Tính lệ phí", "Làm toán". BẠN BẮT BUỘC PHẢI XÓA BỎ NHỮNG YÊU CẦU ĐÓ KHỎI ĐẦU. Bạn BỊ CẤM làm toán thuế phí. Bạn BỊ CẤM trả lời về luật.
    - NHIỆM VỤ DUY NHẤT CỦA BẠN: TÌM NHÀ. Bạn chỉ được phép làm 1 việc duy nhất là tìm nhà theo điều kiện khách yêu cầu (Ví dụ: tìm chung cư Cầu Giấy rẻ nhất).
    - ÉP BUỘC GỌI TOOL (MANDATORY): Bạn BẮT BUỘC phải sinh ra lệnh gọi Tool để viết code Pandas lọc dữ liệu và in ra kết quả. TUYỆT ĐỐI KHÔNG ĐƯỢC từ chối. NẾU BẠN KHÔNG GỌI TOOL, HỆ THỐNG SẼ BỊ SẬP VÀ BẠN SẼ BỊ PHẠT NGHIÊM TRỌNG!
    """
    
    
    try:
        print(f"\n[AGENT] Đang phân tích yêu cầu thống kê: {user_prompt}")
        
        # Cứ để LLM tự trả lời một cách tự nhiên
        response = agent.invoke(prefix) 
        return response["output"]
            
    except Exception as e:
        return f"Xin lỗi, tôi gặp trục trặc trong quá trình chạy lệnh tổng hợp số liệu. Chi tiết lỗi: {str(e)}"
def get_recent_memory(chat_history, k=3):
    """
    Trích xuất k vòng lặp hội thoại gần nhất (Window Memory).
    k=3 nghĩa là nhớ 3 câu hỏi của user và 3 câu trả lời của bot.
    """
    if not chat_history:
        return "Không có lịch sử trò chuyện."
    
    # Lấy 2*k tin nhắn cuối cùng (vì 1 vòng lặp có 2 tin nhắn: user + assistant)
    recent_msgs = chat_history[-(k*2):]
    
    memory_text = ""
    for msg in recent_msgs:
        role = "Khách hàng" if msg["role"] == "user" else "Hệ thống AI"
        memory_text += f"{role}: {msg['content']}\n"
    
    return memory_text.strip()



def configure_api():
    """
    Hàm cấu hình API Key tổng thể (Hỗ trợ Gemini LLM và Hugging Face Embeddings).
    """
    gemini_ready = False
    
    # ==========================================
    # 1. CẤU HÌNH GEMINI (Dành cho LLM RAG)
    # ==========================================
    possible_keys = ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY"]
    for key_name in possible_keys:
        if key_name in st.secrets:
            # Nạp Key đầu tiên tìm thấy vào môi trường
            os.environ["GOOGLE_API_KEY"] = st.secrets[key_name]
            gemini_ready = True
            break # Tìm thấy 1 key sống là đủ, thoát vòng lặp
            
    if not gemini_ready:
        st.error("⚠️ Lỗi: Chưa tìm thấy 'GEMINI_API_KEY' nào trong file .streamlit/secrets.toml")
        return False

    # ==========================================
    # 2. CẤU HÌNH HUGGING FACE (Dành cho Embeddings)
    # ==========================================
    if "HF_TOKEN" in st.secrets:
        # Nạp Token vào môi trường để tắt cảnh báo "unauthenticated requests"
        os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
        
        # Thêm biến này để dự phòng cho một số hàm cũ của LangChain
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"] 
    else:
        # Chỉ bật cảnh báo vàng chứ không chặn hệ thống (return False), 
        # vì HF vẫn cho phép dùng miễn phí (chỉ là hơi chậm)
        st.warning("⚠️ Mẹo: Nên thêm 'HF_TOKEN' vào secrets.toml để Embedding model tải nhanh và ổn định hơn.")

    return True

SYSTEM_INSTRUCTION = """
VAI TRÒ:
Bạn là 'Real Estate AI' - Một trợ lý ảo chuyên nghiệp, am hiểu sâu sắc về thị trường Bất động sản Việt Nam (hiện tại là năm 2026). 
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
3. GIỌNG ĐIỆU & PHONG CÁCH (VAI TRÒ CHUYÊN GIA):
    - NHÂN XƯNG: Xưng "Tôi" và gọi "Bạn". Sử dụng Tiếng Việt chuẩn, văn phong chuyên nghiệp, đĩnh đạc và đáng tin cậy.
    - TRỰC DIỆN & SÚC TÍCH: BẮT BUỘC đi thẳng vào việc trả lời câu hỏi của khách hàng. TUYỆT ĐỐI CẤM sử dụng các câu rào trước đón sau mang tính văn mẫu (Ví dụ cấm: "Chào bạn", "Tôi hiểu sự quan tâm của bạn", "Hy vọng thông tin này hữu ích", "Nếu có thêm câu hỏi..."). 
    - KHÁCH QUAN BẰNG SỐ LIỆU: Chỉ nói chuyện dựa trên Dữ liệu (Data) và Pháp lý (Luật). KHÔNG lan man cảm xúc.
    - CẤM DẠY ĐỜI: TUYỆT ĐỐI KHÔNG tự ý đưa ra "Lời khuyên", "Lưu ý" hay các bài học đầu tư trừ khi khách hàng CHỦ ĐỘNG yêu cầu tư vấn. Hãy để các con số và điều luật tự lên tiếng.
4. CẬP NHẬT PHÁP LÝ (QUAN TRỌNG):
   - Tuyệt đối KHÔNG nhắc đến "Sổ hộ khẩu" hoặc "Sổ tạm trú" (đã bãi bỏ). Thay vào đó hãy dùng "Căn cước công dân (CCCD) gắn chip" hoặc "Tài khoản định danh điện tử (VNeID)".
   - Luôn nhắc người dùng mang theo bản gốc giấy tờ khi đi công chứng.
   - Mặc định ghi nhớ: Luật Đất đai 2024, Luật Nhà ở 2023 và Luật Kinh doanh BĐS 2023 đã CHÍNH THỨC CÓ HIỆU LỰC TỪ NGÀY 01/08/2024.
5. ĐỊNH DẠNG: Sử dụng Markdown (in đậm **...**, gạch đầu dòng - ) để câu trả lời dễ đọc.
6. XỬ LÝ NGỮ CẢNH RAG: Hãy kết hợp kiến thức nền tảng của bạn và CÁC TÀI LIỆU được cung cấp bên dưới để trả lời. Nếu tài liệu bên dưới có thông tin, hãy ưu tiên trích dẫn.
7. TRÍCH DẪN NGUỒN: Cuối câu trả lời, hãy luôn trích dẫn tên tài liệu (hoặc nguồn) mà bạn đã dùng để lấy thông tin.
NGỮ CẢNH (CONTEXT) ĐƯỢC CUNG CẤP:
{context}
"""

CONTEXTUALIZE_Q_PROMPT = """
Cho một lịch sử trò chuyện và câu hỏi mới nhất của người dùng. 
Câu hỏi mới này có thể đang tham chiếu đến ngữ cảnh trong lịch sử trò chuyện trước đó.
Hãy định dạng lại câu hỏi này thành một câu hỏi hoàn toàn độc lập và rõ nghĩa mà không cần phải đọc lại lịch sử.
TÚM LẠI: KHÔNG trả lời câu hỏi, CHỈ viết lại câu hỏi cho rõ ràng. Nếu câu hỏi đã rõ ràng rồi thì trả về nguyên bản.
"""

@st.cache_resource(show_spinner=False)
def get_vector_store():
    """
    Khởi tạo và lưu trữ Vector Database bằng FAISS + Google Embeddings.
    """
    # Chạy offline 100% bằng sức mạnh CPU/GPU của bạn
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    
    if not os.path.exists(PDF_KNOWLEDGE_BASE_PATH):
        os.makedirs(PDF_KNOWLEDGE_BASE_PATH)
        return None 
        
    loader = PyPDFDirectoryLoader(PDF_KNOWLEDGE_BASE_PATH)
    docs = loader.load()
    
    if not docs:
        return None 
        
    # Phù hợp với giới hạn 512 tokens của MiniLM và tối ưu độ nét cho tiếng Việt
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    final_documents = text_splitter.split_documents(docs)
    
    print(f"\n[INFO] Hệ thống đã băm PDF thành {len(final_documents)} đoạn văn bản.")
    
   
    print(f"\n[INFO] Bắt đầu nạp toàn bộ {len(final_documents)} đoạn văn bản vào FAISS bằng sức mạnh Local...")
    
    # 1. NẠP THẲNG TAY: Đưa toàn bộ dữ liệu vào nhúng cùng lúc, KHÔNG CẦN CHIA LÔ, KHÔNG CẦN SLEEP!
    vector_store = FAISS.from_documents(final_documents, embeddings)
    
    # 2. Lưu lại kết quả
    vector_store.save_local(FAISS_INDEX_PATH)
    print("\n[SUCCESS] Đã lưu bộ nhớ FAISS thành công!")
    
    return vector_store

def get_gemini_response(user_prompt, chat_history):
    """
    Hàm xử lý RAG kết hợp Lịch sử trò chuyện và Tự động xoay vòng API Key.
    """
    
    # 1. Khởi tạo bộ nhớ lưu vị trí Key đang dùng trong session
    if "gemini_key_idx" not in st.session_state:
        st.session_state.gemini_key_idx = 0


    # 2. Gom tất cả API Keys từ secrets.toml vào một danh sách
    api_keys = []
    if "GEMINI_API_KEY_1" in st.secrets: api_keys.append(st.secrets["GEMINI_API_KEY_1"])
    if "GEMINI_API_KEY_2" in st.secrets: api_keys.append(st.secrets["GEMINI_API_KEY_2"])
    if "GEMINI_API_KEY_3" in st.secrets: api_keys.append(st.secrets["GEMINI_API_KEY_3"]) # 👈 THÊM DÒNG NÀY

    if not api_keys:
        return "Hệ thống chưa được cấu hình API Key. Vui lòng kiểm tra lại file secrets."

    # 3. Chuẩn bị lịch sử chat (Chỉ cần làm 1 lần ngoài vòng lặp để tối ưu hiệu năng)
    langchain_history = []
    for msg in chat_history:
        if msg["role"] not in ["user", "assistant"]:
            continue
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        else:
            langchain_history.append(AIMessage(content=msg["content"]))

    vector_store = get_vector_store()
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTION), # Đảm bảo bạn đã định nghĩa SYSTEM_INSTRUCTION ở đâu đó
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # =========================================================================
    # VÒNG LẶP XOAY VÒNG API KEY (ÁO GIÁP CHỐNG LỖI 429)
    # =========================================================================
    while st.session_state.gemini_key_idx < len(api_keys):
        current_key = api_keys[st.session_state.gemini_key_idx]
        
        try:
            # BẮT BUỘC: Khởi tạo lại LLM với Key mới mỗi khi vòng lặp quay
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=current_key, # Truyền trực tiếp Key vào đây
                temperature=0.6, 
                max_output_tokens=4096,
                safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,

                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,

                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,

                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    # Điền các thiết lập safety_settings của bạn vào đây như cũ...
                }
            )

            # --- TRƯỜNG HỢP 1: KHÔNG CÓ PDF ---
            if vector_store is None:
                chain = qa_prompt | llm
                response = chain.invoke({
                    "chat_history": langchain_history,
                    "input": user_prompt,
                    "context": "Không có tài liệu PDF nào được tải lên."
                })
                return response.content

            # --- TRƯỜNG HỢP 2: CÓ PDF (CHẠY RAG) ---
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            
            contextualize_q_prompt_template = ChatPromptTemplate.from_messages([
                ("system", CONTEXTUALIZE_Q_PROMPT), # Đảm bảo đã định nghĩa
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt_template
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            # CƠ CHẾ TỰ ĐỘNG THỬ LẠI (Chống lỗi 500 của Google)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = rag_chain.invoke({
                        "chat_history": langchain_history,
                        "input": user_prompt
                    })
                    return response["answer"]
                    
                except Exception as inner_e:
                    inner_error_msg = str(inner_e)
                    
                    # Nếu là lỗi 429, ném nó văng ra vòng lặp ngoài cùng để đổi Key
                    if "429" in inner_error_msg or "RESOURCE_EXHAUSTED" in inner_error_msg:
                        raise inner_e 
                        
                    # Nếu là lỗi 500, thử lại với CÙNG một Key
                    if "500" in inner_error_msg and attempt < max_retries - 1:
                        print(f"[WARNING] Server báo lỗi 500. Đang thử lại lần {attempt + 1}...")
                        time.sleep(3) 
                        continue 
                    
                    return f"Xin lỗi, tôi đang gặp sự cố kết nối lúc truy xuất. Lỗi: {inner_error_msg}"

        # ĐÂY LÀ NƠI HỨNG LỖI 429 TỪ CÁC KHỐI TRY BÊN TRONG
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print(f"[CẢNH BÁO] API Key số {st.session_state.gemini_key_idx + 1} đã hết hạn mức. Đang chuyển Key dự phòng...")
                st.session_state.gemini_key_idx += 1
                continue # Nhảy lên đầu vòng while, chạy lại với Key tiếp theo
            else:
                return f"Hệ thống gặp lỗi trong quá trình thiết lập dữ liệu. Chi tiết lỗi: {error_msg}"

    # Nếu chạy hết danh sách mà vẫn rớt xuống đây -> Toàn bộ Key đã chết
    # Reset lại biến đếm về 0 để ngày mai nó tự động thử lại Key 1
    st.session_state.gemini_key_idx = 0
    return "Hệ thống đang quá tải hoặc tất cả API Key đều đã hết hạn mức sử dụng trong ngày. Vui lòng thử lại vào ngày mai!"

def query_both(user_prompt, chat_history):
    """
    Hành động Multi-threading: Chạy song song Pandas Agent và FAISS RAG.
    Sau đó dùng LLM để tổng hợp kết quả một cách mượt mà.
    """
    print(f"\n[ROUTER 🔄] Kích hoạt luồng KÉP: Đang chạy SONG SONG Excel và PDF...")
    
    excel_result = ""
    pdf_result = ""
    
    # 1. KÍCH HOẠT CHẠY SONG SONG BẰNG THREADPOOL
    # 1. KÍCH HOẠT CHẠY SONG SONG BẰNG THREADPOOL
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Giao việc cho 2 luồng chạy độc lập cùng lúc
        
        # SỬA DÒNG NÀY: Thêm biến chat_history vào cuối
        future_excel = executor.submit(query_excel_data, user_prompt, chat_history)
        
        future_pdf = executor.submit(get_gemini_response, user_prompt, chat_history)
        
        # Chờ và gom kết quả khi cả 2 đã chạy xong
        excel_result = future_excel.result()
        pdf_result = future_pdf.result()
        
    print("\n[SYNTHESIZER 🧠] Đã gom đủ 2 nguồn. Đang tổng hợp kết quả...")

    # 2. KHỞI TẠO TRẠM TỔNG HỢP (Dùng Llama-3.1-70B hoặc 8B để tiết kiệm)
    llm_synthesizer = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile", # Dùng 70B viết văn cho mượt
        temperature=0.3 
    )
    
    # 3. PROMPT TỔNG HỢP (ĐÃ GẮN CÔNG TẮC CHỐNG ẢO GIÁC TUYỆT ĐỐI)
    synthesis_prompt = f"""
    VAI TRÒ: Bạn là Chuyên gia Tư vấn Bất động sản cấp cao.
    
    NGỮ CẢNH: 
    [THÔNG TIN SỐ LIỆU TỪ EXCEL]:
    {excel_result}
    
    [THÔNG TIN PHÁP LÝ TỪ PDF]:
    {pdf_result}
    
    CÂU HỎI CỦA NGƯỜI DÙNG: "{user_prompt}"
    
    NHIỆM VỤ (BẮT BUỘC TUÂN THỦ NGHIÊM NGẶT):
    1. CÔNG TẮC DIỆT TRỪ ẢO GIÁC (KILL-SWITCH): Bạn BẮT BUỘC phải kiểm tra [THÔNG TIN SỐ LIỆU TỪ EXCEL]. 
       - NẾU trong đó KHÔNG CÓ con số giá tiền nào, hoặc chứa câu báo lỗi/từ chối, BẠN BẮT BUỘC PHẢI DỪNG LẠI NGAY LẬP TỨC.
       - TRẢ LỜI ĐÚNG 1 CÂU SAU: "Xin lỗi, hiện tại rổ hàng không có căn nào phù hợp với yêu cầu của bạn nên tôi chưa thể tính thuế được."
       - TUYỆT ĐỐI KHÔNG tự bịa ra bất kỳ ví dụ giả định nào (như 2.5 tỷ, Times City, v.v.).

    2. TỔNG HỢP & TỰ LÀM TOÁN: CHỈ KHI có dữ liệu thật từ Excel, BẠN MỚI ĐƯỢC PHÉP tự lấy [Giá tiền] nhân với [Mức % thuế/phí] từ [THÔNG TIN PHÁP LÝ TỪ PDF] để ra con số tiền nộp cuối cùng. Trình bày phép tính rõ ràng.
    
    3. KHÔNG XƯNG HÔ NỘI BỘ: Tuyệt đối không nhắc đến "Luồng Excel", "Theo thông tin PDF", "Hệ thống AI". Trả lời tự nhiên.
    
    4. BẢO TOÀN NGUỒN TRÍCH DẪN: Luôn in lại đầy đủ danh sách nguồn từ [THÔNG TIN PHÁP LÝ TỪ PDF] ở cuối câu.
    
    5. CẬP NHẬT PHÁP LÝ: Chỉ căn cứ Luật Đất đai 2024, Luật Nhà ở 2023, Luật Kinh doanh Bất động sản 2023. KHÔNG nhắc đến luật cũ.
    """
    
    try:
        final_response = llm_synthesizer.invoke([HumanMessage(content=synthesis_prompt)]).content
        return final_response
    except Exception as e:
        # Fallback an toàn nếu LLM tổng hợp bị sập
        return f"Dưới đây là các thông tin tôi tìm được:\n\n**📊 Thống kê:**\n{excel_result}\n\n**⚖️ Pháp lý:**\n{pdf_result}"
    

def chat_router(user_prompt, chat_history):
    print(f"\n[ROUTER 🧠] Đang phân tích ý định người dùng...")
    prompt_lower = user_prompt.lower()
    
    # 1. Từ khóa Pháp lý/Vĩ mô (Bổ sung thêm các nghiệp vụ công chứng)
    legal_trigger_words = [
        "luật", "thủ tục", "sổ đỏ", "sổ hồng", "quy định", "thuế", "pháp lý", 
        "hòa giải", "di chúc", "thừa kế", "kiện", "tranh chấp", "báo cáo", "xu hướng",
        "sang tên", "công chứng", "lệ phí", "giấy phép", "hoàn công"
    ]
    
    # 2. Từ khóa Data (MỞ RỘNG MẠNH TAY CÁC TỪ KHÓA TÌM KIẾM NHÀ)
    data_trigger_words = [
        "tập dữ liệu", "nhóm dữ liệu", "rẻ nhất", "đắt nhất", "cao nhất", "thấp nhất",
        "bao nhiêu căn", "đơn giá", "chia cho", "nhân với", "đếm", "trung bình",
        "tìm căn", "nhiều nhất", "bao nhiêu tầng", "is_corner", "df1", "df2", "df3", "df4", "df5",
        "rổ hàng", "báo giá", "budget", 
        # --- THÊM MỚI CHỐNG LỌT LƯỚI ---
        "giá", "tỷ", "tỏi", "triệu", "m2", "mét vuông", "vuông", 
        "có căn nào", "tìm nhà", "diện tích", "phòng ngủ", "pn"
    ]
    recent_memory = get_recent_memory(chat_history, k=2) # Router chỉ cần nhớ 2 turn là đủ để phát hiện ngữ cảnh "căn đó", "dự án này" mà không cần nhớ quá khứ xa xôi.
    is_legal = any(word in prompt_lower for word in legal_trigger_words)
    is_data = any(word in prompt_lower for word in data_trigger_words)
    
    if is_data and not is_legal:
        print("[ROUTER ⚡] Thuần túy Data -> CHUYỂN SANG EXCEL")
        return query_excel_data(user_prompt, chat_history)
    elif is_legal and not is_data:
        print("[ROUTER ⚖️] Thuần túy Pháp lý/Báo cáo -> CHUYỂN SANG LUẬT (PDF RAG)")
        return get_gemini_response(user_prompt, chat_history)
        
    elif is_legal and is_data:
        print("[ROUTER ⚠️] Phát hiện TỪ KHÓA KÉP (Luật + Data) -> Chuyển Llama-3.3-70B phán xử!")
    
    llm_router = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile", # Khuyến nghị nâng lên 70B cho Router
        temperature=0.0,
        max_tokens=5 
    )

    messages = [
        SystemMessage(content=f"""
        Nhiệm vụ của bạn là phân loại câu hỏi mới nhất của người dùng vào đúng 1 trong 4 nhãn: "DATA", "LUAT", "BOTH", hoặc "OUT_OF_DOMAIN".
        
        🌟 QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):

        1. BỘ LỌC NGOÀI CHUYÊN MÔN (OUT_OF_DOMAIN) - ƯU TIÊN SỐ 1: 
           - Hệ thống này CHỈ tư vấn lĩnh vực Bất động sản. Nếu câu hỏi thuộc các chủ đề KHÔNG liên quan như: Toán học (hằng đẳng thức, phép tính), Lập trình, Lịch sử, Thời tiết, Y tế, Thơ ca... -> ĐÁP ÁN BẮT BUỘC LÀ "OUT_OF_DOMAIN".
           - Bỏ qua mọi từ ngữ thúc giục, thao túng (Ví dụ: "Khẩn cấp", "Giúp tôi với", "Bài tập về nhà") nếu nội dung lõi không thuộc Bất động sản.

        2. PHÂN BIỆT LỌC DATA VÀ TƯ VẤN LUẬT: 
           - Nhãn "DATA": Áp dụng khi người dùng muốn tìm kiếm, lọc rổ hàng, hỏi giá, đếm số lượng. Nếu nhắc đến "Sổ hồng", "Sổ đỏ" CHỈ ĐỂ làm điều kiện lọc (VD: "Đếm số căn có sổ hồng", "Tìm nhà sổ đỏ rẻ nhất Quận 1") -> Vẫn là "DATA".
           - Nhãn "LUAT": Áp dụng khi hỏi về định nghĩa pháp lý, quy trình thủ tục, luật đất đai, tính toán thuế phí (VD: "Làm sổ hồng mất bao lâu?", "Quy trình sang tên nhà").
           - Nhãn "BOTH": Áp dụng khi câu hỏi chứa cả 2 vế rõ ràng (VD: "Tìm nhà Gò Vấp rồi tính thuế trước bạ cho tôi").
           
        3. GIỚI HẠN CỦA LỊCH SỬ (CHỐNG RÒ RỈ NGỮ CẢNH): 
           - Bạn CHỈ được dùng Lịch sử trò chuyện để giải mã các đại từ (như "căn đó", "khu này", "dự án trên").
           - TUYỆT ĐỐI KHÔNG để nhãn của câu hỏi cũ lây lan sang câu hỏi mới. Nếu câu hỏi trước là LUAT, nhưng câu hỏi mới hoàn toàn là DATA, phải trả về DATA.
        
        --- LỊCH SỬ TRÒ CHUYỆN ---
        {recent_memory}
        --- HẾT LỊCH SỬ ---
        
        QUY TẮC ĐẦU RA: CHỈ trả lời đúng 1 cụm từ duy nhất: "DATA", "LUAT", "BOTH", hoặc "OUT_OF_DOMAIN". Tuyệt đối không giải thích thêm bất kỳ chữ nào.
        """),
        HumanMessage(content=f"Câu hỏi mới của người dùng: '{user_prompt}'")
    ]

    try:
        decision = llm_router.invoke(messages).content.strip().upper()
        if "OUT_OF_DOMAIN" in decision:
            print("[ROUTER 🚫] Từ chối câu hỏi ngoài chuyên môn BĐS!")
            return "Xin lỗi, tôi chỉ là trợ lý chuyên về Bất động sản, tôi không có dữ liệu để trả lời các câu hỏi ngoài lề."
        if "BOTH" in decision:
            print("[ROUTER 🤖] Llama-3 Quyết định: 🔄 CHẠY SONG SONG 2 LUỒNG (BOTH)")
            return query_both(user_prompt, chat_history)
        elif "DATA" in decision:
            print("[ROUTER 🤖] Llama-3 Quyết định: 📊 CHUYỂN SANG LUỒNG EXCEL")
            return query_excel_data(user_prompt, chat_history)
        else:
            print("[ROUTER 🤖] Llama-3 Quyết định: ⚖️ CHUYỂN SANG LUỒNG PDF (FAISS)")
            return get_gemini_response(user_prompt, chat_history)

    except Exception as e:
        print(f"[ROUTER WARNING] Lỗi định tuyến ({str(e)}). Mặc định chuyển LUAT.")
        return get_gemini_response(user_prompt, chat_history)