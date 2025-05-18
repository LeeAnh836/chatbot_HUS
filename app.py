import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Tải dữ liệu từ file JSON ---
DATA_PATH = "data.json"  # Sử dụng đường dẫn tương đối

try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        scripts = json.load(f)
except Exception as e:
    st.error(f"❌ Không thể tải file JSON: {e}")
    st.stop()

# --- Trích xuất câu hỏi và câu trả lời ---
questions = []
answers = []

if isinstance(scripts, list):
    if isinstance(scripts[0], dict) and "question" in scripts[0] and "answer" in scripts[0]:
        questions = [item["question"] for item in scripts]
        answers = [item["answer"] for item in scripts]
    else:
        st.error("⚠️ Dữ liệu JSON không đúng định dạng mong muốn.")
        st.stop()
else:
    st.error("⚠️ File JSON phải chứa danh sách các đoạn hội thoại.")
    st.stop()

# Kiểm tra dữ liệu đã tải
if not questions:
    st.warning("⚠️ Không tìm thấy dữ liệu hội thoại phù hợp trong file JSON.")
    st.stop()

# --- Vector hóa dữ liệu bằng TF-IDF ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# --- Giao diện Streamlit ---
st.set_page_config(page_title="🎓 Chatbot Tư vấn Tuyển sinh", layout="wide")
st.title("🎓 Chatbot Tư vấn Tuyển sinh")

# Khởi tạo session state
if "history" not in st.session_state:
    st.session_state.history = []

# Giao diện nhập liệu chính
col1, col2 = st.columns([3, 1])

with col1:
    # Hiển thị lịch sử chat trong khung chat chính
    chat_container = st.container()
    with chat_container:
        for q, a in st.session_state.history:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                <p style='margin: 0; color: #222;'><strong>🧑‍🎓 Bạn:</strong> {q.replace(chr(10), '<br>')}</p>
            </div>
            <div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                <p style='margin: 0; color: #1565c0;'><strong>🤖 Chatbot:</strong> {a.replace(chr(10), '<br>')}</p>
            </div>
            """, unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Nhập câu hỏi của bạn:")
        submit = st.form_submit_button("Gửi")

    if submit and user_input:
        user_vec = vectorizer.transform([user_input])
        sims = cosine_similarity(user_vec, X)
        best_idx = sims.argmax()
        best_score = sims[0, best_idx]

        if best_score > 0.3:
            reply = answers[best_idx]
        else:
            reply = "❗Xin lỗi, mình chưa có thông tin để trả lời câu hỏi này."

        # Lưu lịch sử hội thoại
        st.session_state.history.append((user_input, reply))
        # Reload trang để hiển thị tin nhắn mới
        st.rerun()

# Hiển thị lịch sử ở Sidebar
with st.sidebar:
    st.header("📜 Lịch sử Chat gần đây")
    if st.session_state.history:
        # Chỉ hiển thị 5 tin nhắn gần nhất
        recent_history = st.session_state.history[-5:]
        for q, a in recent_history:
            with st.expander(f"Q: {q[:30]}..." if len(q) > 30 else f"Q: {q}"):
                st.markdown(f"**A:** {a}")
    else:
        st.info("Chưa có lịch sử chat")
