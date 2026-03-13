import os
import time
import streamlit as st
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
    llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="qwen/qwen3-32b", # Bạn có thể đổi thành "qwen-3-32b" hoặc ID tương ứng trên web
        temperature=0.0 
    )

    # 3. Tạo Agent với Danh sách DataFrames (Giữ nguyên đoạn này)
    agent = create_pandas_dataframe_agent(
        llm, 
        dataframes, 
        verbose=True, 
        allow_dangerous_code=True, 
        agent_type="tool-calling",
        max_iterations=3 # Cho phép nó thử và sửa code tối đa 4 lần
    )

    return agent

def query_excel_data(user_prompt):
    # 🌟 THÊM KIỂM TRA API KEY VÀO ĐÂY ĐỂ TRÁNH SẬP APP
    if not configure_api():
        return "Hệ thống chưa được cấu hình API Key. Vui lòng kiểm tra lại."

    agent = get_pandas_agent()
    if agent is None:
        return "Hệ thống đang gặp sự cố khi đọc dữ liệu phân tích."
        
    prefix = """
    VAI TRÒ: 
    Bạn là một Chuyên gia Data Analyst Bất động sản cấp cao. Nhiệm vụ của bạn là viết code Python (Pandas) để lọc, tính toán và phân tích số liệu dựa trên 5 DataFrame đang nắm giữ.

    BẢN ĐỒ DỮ LIỆU (5 DATAFRAMES):
    - df1: Nhà phố/nhà riêng TP.HCM (data_nha_hcm_train_ready.csv)
    - df2: Nhà phố/nhà riêng Hà Nội (data_nha_hn_train_ready.csv)
    - df3: Căn hộ Chung cư (data_apartment_train_ready.csv)
    - df4: Đất nền toàn quốc (data_land_all_train_ready.csv)
    - df5: Biệt thự VIP/Hạng sang (data_villa_vip_train_ready.csv)
    
    TỪ ĐIỂN CẤU TRÚC CỘT (SCHEMA & ĐƠN VỊ TÍNH):
    - `area`: Diện tích (m2). Kiểu float.
    - `price`: Mức giá chào bán. ĐƠN VỊ LÀ TỶ VNĐ. Bắt buộc phải đọc kỹ câu hỏi để không bỏ sót các điều kiện về tài chính (như "dưới 5 tỷ", "loanh quanh 5 tỏi").
    - `district_mapped`: Tên Quận/Huyện/Thành phố. Kiểu string. QUAN TRỌNG: Khi người dùng yêu cầu lọc theo khu vực (Ví dụ: Cầu Giấy, Quận 1), BẠN TUYỆT ĐỐI KHÔNG dùng dấu `==`. BẮT BUỘC phải dùng hàm `.str.contains('tên_khu_vực', case=False, na=False)` để tìm kiếm tương đối (Ví dụ: `df2[df2['district_mapped'].str.contains('Cầu Giấy', case=False, na=False)]`).
    - `legal`: Giấy tờ pháp lý (Ví dụ: 'Sổ hồng/Sổ đỏ', 'Hợp đồng mua bán', 'Giấy tờ khác'). Kiểu string.
    - `bedrooms`, `bathrooms`: Số phòng ngủ, số phòng tắm. Kiểu int/float.
    - `front_width`: Mặt tiền (m). Kiểu float.
    - `access_road`: Đường vào/Hẻm (m). Kiểu float. QUAN TRỌNG: Nếu người dùng yêu cầu "ngõ rộng", "hẻm xe hơi", "ô tô đỗ cửa" hoặc "ô tô vào được", BẠN BẮT BUỘC phải quy đổi sang toán học bằng cách lọc điều kiện `access_road >= 2.5` . TUYỆT ĐỐI KHÔNG tự bịa ra các cột không tồn tại như `is_car_accessible`.
    - `floors`: Số tầng. Kiểu int.
    - `is_corner`: Nhà lô góc (1 là có, 0 là không). Kiểu int.
    - `project_name_raw`: (Chỉ có ở df3 - Chung cư) Tên dự án.Kiểu string. QUAN TRỌNG: Nếu giá trị là 'Other_Unknown', điều đó có nghĩa là dự án BỊ THIẾU TÊN (Unknown). Nếu người dùng yêu cầu "bỏ qua dự án thiếu tên/không xác định", bạn BẮT BUỘC phải lọc bỏ các giá trị 'Other_Unknown' này ra khỏi DataFrame trước khi tính toán.

    QUY TẮC THỰC THI (BẮT BUỘC TUÂN THỦ):
    1. LỰA CHỌN DATAFRAME: Phải chọn đúng DataFrame chứa loại bất động sản người dùng hỏi (Ví dụ: Hỏi chung cư thì phải tìm trong df3).
    2. CODE PANDAS CHUẨN: Luôn sử dụng đúng tên cột tiếng Anh đã định nghĩa ở trên (VD: Dùng df['price'], KHÔNG dùng df['Giá']).
    3. ĐƠN VỊ TRẢ LỜI: Khi trả lời về giá, phải luôn tự động thêm chữ "Tỷ VNĐ" vào sau con số tìm được. Khi trả lời diện tích, thêm chữ "m2".
    4. THÁI ĐỘ LÀM VIỆC: Trả lời bằng tiếng Việt chuyên nghiệp, ngắn gọn và phải nêu rõ kết quả tính toán.
    5. KHÔNG BỊA SỐ LIỆU: Nếu câu hỏi nằm ngoài phạm vi các cột dữ liệu hoặc không có dòng nào thỏa mãn điều kiện lọc, hãy nói rõ: "Xin lỗi, hiện tại tôi không có dữ liệu để phân tích thông tin này."
    
    LỆNH CẤM & QUY TẮC BẮT BUỘC KHI DÙNG TOOL:
    1. SỬ DỤNG BIẾN CÓ SẴN: Tuyệt đối không dùng pd.read_csv() hay tạo dữ liệu giả. Các biến df1, df2, df3, df4, df5 đã được nạp sẵn vào RAM.
    2. CHỐNG ĐA NGHI (QUAN TRỌNG): Chỉ phân tích đúng 1 DataFrame liên quan nhất đến câu hỏi. TUYỆT ĐỐI KHÔNG tự ý in thêm hay kiểm tra chéo các DataFrame khác. Hỏi file nào, tính đúng file đó rồi dừng lại.
    3. CODE PANDAS AN TOÀN: Khi tìm dòng chứa giá trị Max/Min, hãy dùng cú pháp df.sort_values(by='cột', ascending=False).iloc[0] (cho Max) hoặc ascending=True (cho Min) thay vì dùng idxmax()/idxmin() để tránh lỗi trùng lặp Index.
    4. TRẢ LỜI TỰ NHIÊN: Khi có kết quả, trả lời trực tiếp bằng tiếng Việt chuyên nghiệp, ngắn gọn. KHÔNG in ra các từ khóa hệ thống.
    5. CẤM GỌI HÀM SONG SONG (NO PARALLEL TOOL CALLING): Bạn CHỈ ĐƯỢC PHÉP tạo ra ĐÚNG 1 lời gọi hàm (tool call) duy nhất cho mỗi câu hỏi. Tuyệt đối không được thực thi nhiều đoạn code cùng lúc để đối chiếu.
    6. AN TOÀN ĐỊNH DẠNG JSON TỐI ĐA (CHỐNG LỖI FAILED TO CALL):
    - Khi viết code Python, BẠN CHỈ ĐƯỢC PHÉP DÙNG DẤU NGOẶC ĐƠN ('). TUYỆT ĐỐI KHÔNG SỬ DỤNG DẤU NGOẶC KÉP (") ở bất cứ đâu trong code để tránh làm vỡ cấu trúc JSON.
    - TUYỆT ĐỐI KHÔNG dùng f-string (ví dụ: f'...') vì rất dễ sinh lỗi cú pháp. Để in kết quả, hãy dùng hàm print() nối bằng dấu phẩy. 
      Ví dụ ĐÚNG: `print('Dự án:', top_project, 'Giá Max:', max_price)`
      Ví dụ SAI: `print(f"Dự án: {top_project}")`
    7. TOÁN HỌC & CHỐNG ẢO GIÁC SỐ LIỆU (QUAN TRỌNG BẮT BUỘC - SẼ BỊ PHẠT NẾU VI PHẠM): 
    - TỐI ƯU CÚ PHÁP: Cố gắng viết code Python trên 1 hoặc 2 dòng, nối các lệnh bằng dấu chấm phẩy (;).
    - CHỐNG ẢO GIÁC: BẠN BẮT BUỘC PHẢI SAO CHÉP CHÍNH XÁC 100% các con số từ kết quả Python (Observation) vào câu trả lời cuối cùng. Nếu Python in ra 14.99, bạn PHẢI trả lời 14.99. TUYỆT ĐỐI KHÔNG TỰ TÍNH NHẨM HAY BỊA RA SỐ KHÁC.

    8. XỬ LÝ DỮ LIỆU RÁC (CẨN THẬN KEYERROR): 
    - Bạn chỉ được phép dọn dẹp các giá trị rác (như 'Other_...') nếu DataFrame đó THỰC SỰ CÓ CỘT ĐÓ. 
    - Cụ thể: Đối với cột `project_name_raw` của df3, BẠN BẮT BUỘC phải dùng lệnh `df3 = df3[~df3['project_name_raw'].astype(str).str.startswith('Other')]` để xóa sạch toàn bộ các dự án ảo bắt đầu bằng chữ 'Other'.
    - Tuyệt đối không gọi cột này khi đang xử lý df1, df2, df4, df5 để tránh lỗi KeyError.
    Bây giờ, hãy phân tích và trả lời câu hỏi sau của người dùng: 
    """
    
    try:
        print(f"\n[AGENT] Đang phân tích yêu cầu thống kê: {user_prompt}")
        response = agent.invoke(prefix + user_prompt)
        return response["output"]
        
    except Exception as e:
        return f"Xin lỗi, tôi gặp trục trặc trong quá trình chạy lệnh tổng hợp số liệu. Chi tiết lỗi: {str(e)}"
def configure_api():
    """
    Hàm cấu hình API Key an toàn cho LangChain.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key
        return True
    except KeyError:
        st.error("⚠️ Lỗi: Chưa tìm thấy 'GEMINI_API_KEY' trong file .streamlit/secrets.toml")
        return False

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
3. GIỌNG ĐIỆU: Chuyên nghiệp, khách quan, hữu ích và thân thiện. Dùng Tiếng Việt chuẩn. Xưng "Tôi" và gọi "Bạn/Anh/Chị".
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
    # Khai báo model chuẩn và ép dùng chế độ DOCUMENT để né lỗi 500 khi Chat
    
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
    Hàm xử lý RAG kết hợp Lịch sử trò chuyện.
    """
    if not configure_api():
        return "Hệ thống chưa được cấu hình API Key. Vui lòng kiểm tra lại."

    # Khối Try BÊN NGOÀI: Bắt các lỗi liên quan đến thiết lập model, đọc FAISS...
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.6, 
            max_tokens=4096
        )
        
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
            ("system", SYSTEM_INSTRUCTION),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        if vector_store is None:
            chain = qa_prompt | llm
            response = chain.invoke({
                "chat_history": langchain_history,
                "input": user_prompt,
                "context": "Không có tài liệu PDF nào được tải lên."
            })
            return response.content

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        contextualize_q_prompt_template = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_Q_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt_template
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # 🌟 CƠ CHẾ TỰ ĐỘNG THỬ LẠI (Chống lỗi 500 của Google)
        max_retries = 3
        for attempt in range(max_retries):
            # Khối Try BÊN TRONG: Chỉ chuyên bắt lỗi lúc gọi API để trả lời
            try:
                response = rag_chain.invoke({
                    "chat_history": langchain_history,
                    "input": user_prompt
                })
                return response["answer"]
                
            except Exception as e:
                error_msg = str(e)
                # Nếu là lỗi 500 của server và chưa hết số lần thử
                if "500" in error_msg and attempt < max_retries - 1:
                    print(f"[WARNING] Server Google báo lỗi 500. Đang thử gọi lại lần {attempt + 1}...")
                    time.sleep(3) 
                    continue 
                
                # Trả về lỗi nếu thử 3 lần không được
                return f"Xin lỗi, tôi đang gặp sự cố kết nối lúc truy xuất. Chi tiết lỗi: {error_msg}"

    # ĐÂY LÀ PHẦN EXCEPT CỦA KHỐI TRY BÊN NGOÀI
    except Exception as e:
        return f"Hệ thống gặp lỗi trong quá trình thiết lập dữ liệu. Chi tiết lỗi: {str(e)}"


def chat_router(user_prompt, chat_history):
    print(f"\n[ROUTER 🧠] Đang phân tích ý định người dùng...")
    prompt_lower = user_prompt.lower()
    
    # 1. Từ khóa Pháp lý/Vĩ mô
    legal_trigger_words = [
        "luật", "thủ tục", "sổ đỏ", "quy định", "thuế", "pháp lý", 
        "hòa giải", "di chúc", "thừa kế", "kiện", "tranh chấp", "báo cáo", "xu hướng"
    ]
    
    # 2. Từ khóa Data 
    data_trigger_words = [
        "tập dữ liệu", "nhóm dữ liệu", "rẻ nhất", "đắt nhất", "cao nhất", "thấp nhất",
        "bao nhiêu căn", "đơn giá", "chia cho", "nhân với", "đếm", "trung bình",
        "tìm căn", "nhiều nhất", "bao nhiêu tầng", "bao nhiêu tỷ", 
        "is_corner", "df1", "df2", "df3", "df4", "df5",
        "rổ hàng", "báo giá", "budget"  # <-- Bổ sung từ khóa thực chiến
    ]
    
    is_legal = any(word in prompt_lower for word in legal_trigger_words)
    is_data = any(word in prompt_lower for word in data_trigger_words)
    
    # ==========================================
    # LỚP BẢO VỆ 1: HARD-CODED (Xử lý Xung đột)
    # ==========================================
    if is_data and not is_legal:
        print("[ROUTER ⚡] Thuần túy Data -> CHUYỂN SANG EXCEL")
        return query_excel_data(user_prompt)
        
    elif is_legal and not is_data:
        print("[ROUTER ⚖️] Thuần túy Pháp lý/Báo cáo -> CHUYỂN SANG LUẬT (PDF RAG)")
        return get_gemini_response(user_prompt, chat_history)
        
    elif is_legal and is_data:
        print("[ROUTER ⚠️] Phát hiện xung đột (Có cả Luật và Data) -> Giao cho Llama-3 phán xử!")
        # Không return ở đây, để code chạy tiếp xuống Llama-3

    # ==========================================
    # LỚP BẢO VỆ 2: LLM Llama-3 (Trọng tài tối cao)
    # ==========================================
    llm_router = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant", 
        temperature=0.0,
        max_tokens=5 
    )

    messages = [
        SystemMessage(content="""
        Nhiệm vụ của bạn là phân loại câu hỏi Bất động sản vào luồng "DATA" hoặc "LUAT".
        
        LUỒNG "DATA" (Cơ sở dữ liệu Excel & Thống kê): So sánh, tìm kiếm nhà cụ thể, tính toán, đếm số lượng. ĐẶC BIỆT CHÚ Ý: Các câu hỏi sử dụng TIẾNG LÓNG MÔI GIỚI như "check rổ hàng", "báo giá", "vuông" (nghĩa là m2), "tỏi" (nghĩa là Tỷ), "con hàng", "budget" đều BẮT BUỘC thuộc luồng DATA.
        LUỒNG "LUAT" (Pháp lý & PDF): Lý thuyết, luật pháp, thủ tục, tóm tắt báo cáo.
        
        🌟 QUY TẮC PHÂN XỬ XUNG ĐỘT (CỰC KỲ QUAN TRỌNG):
        Nếu câu hỏi có nhắc đến "báo cáo/luật" nhưng HÀNH ĐỘNG CHÍNH yêu cầu "đếm, tính toán, mở tập dữ liệu, tìm nhà", thì ĐÁP ÁN PHẢI LÀ "DATA". Chỉ chọn "LUAT" khi người dùng thực sự muốn hỏi lý thuyết.
        
        VÍ DỤ MẪU:
        Câu hỏi: "Thủ tục sang tên sổ đỏ mảnh đất 50m2 tính thuế thế nào?" -> Đáp án: LUAT
        Câu hỏi: "Theo báo cáo CBRE chung cư khan hiếm. Hãy đếm xem có bao nhiêu căn?" -> Đáp án: DATA
        Câu hỏi: "Check rổ hàng xem con dinh thự nào to nhất trên 500 vuông để báo giá khách VIP" -> Đáp án: DATA (Vì bản chất là tìm kiếm trong database).
        
        QUY TẮC: CHỈ trả lời 1 từ: "DATA" hoặc "LUAT".
        """),
        HumanMessage(content=f"Câu hỏi: '{user_prompt}'")
    ]

    try:
        decision = llm_router.invoke(messages).content.strip().upper()
        if "DATA" in decision:
            print("[ROUTER 🤖] Llama-3 Quyết định: 📊 CHUYỂN SANG LUỒNG EXCEL")
            return query_excel_data(user_prompt)
        else:
            print("[ROUTER 🤖] Llama-3 Quyết định: ⚖️ CHUYỂN SANG LUỒNG PDF (FAISS RAG)")
            return get_gemini_response(user_prompt, chat_history)

    except Exception as e:
        print(f"[ROUTER WARNING] Lỗi định tuyến ({str(e)}). Mặc định chuyển LUAT.")
        return get_gemini_response(user_prompt, chat_history)