# Chatbot Tư vấn Tuyển sinh

Đây là một chatbot được xây dựng bằng Streamlit để tư vấn tuyển sinh. Chatbot sử dụng phương pháp TF-IDF và cosine similarity để tìm câu trả lời phù hợp nhất cho câu hỏi của người dùng.

## Cài đặt

1. Clone repository này về máy local
2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```
3. Chạy ứng dụng:
```bash
streamlit run app.py
```

## Cấu trúc dự án

- `app.py`: File chính chứa code của chatbot
- `data.json`: File dữ liệu chứa câu hỏi và câu trả lời
- `requirements.txt`: Danh sách các thư viện cần thiết 