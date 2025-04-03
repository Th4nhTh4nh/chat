import streamlit as st
import google.generativeai as genai
import os
import uuid
from dotenv import load_dotenv
import tempfile
from datetime import datetime
from sqlite import *
load_dotenv()


# Cấu hình chính
def configure_gemini_api():
    genai.configure(api_key=os.getenv("API_KEY"))
    return genai.GenerativeModel(os.getenv("MODEL_NAME"))


# Ghi nhỡ ngữ cảnh hội thoại
def multi_conversation(model, conv_id=None):
    # Nếu là cuộc trò chuyện mới hoặc không có lịch sử, tạo chat mới
    if conv_id is None or conv_id not in st.session_state.chats:
        chat = model.start_chat(history=[])

        # Lưu chat vào session state nếu là cuộc trò chuyện đã được lưu
        if conv_id:
            st.session_state.chats[conv_id] = chat

        return chat

    # Nếu đã có chat trong session state, trả về chat đó
    return st.session_state.chats[conv_id]


# Tạo câu trả lời với tệp
def generate_response(chat, prompt, pdf_file=None):
    try:
        if pdf_file:
            # Lưu file tải lên vào một file tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name

            # Sử dụng đường dẫn file với genai API
            with open(tmp_file_path, 'rb') as f:
                file_content = f.read()

            # Tạo một phần cho file PDF
            pdf_part = {"mime_type": "application/pdf", "data": file_content}

            # Gửi tin nhắn với cả văn bản và PDF
            response = chat.send_message([prompt, pdf_part])

            # Xóa file tạm
            os.unlink(tmp_file_path)

            return response.text
        else:
            response = chat.send_message(prompt)
            return response.text
    except Exception as e:
        return f"Error: {str(e)}"


def display_message(role, content):
    with st.chat_message(role):
        st.write(content)


# Lưu hội thoại
def save_conversation(title, messages, conv_id=None):
    # Lưu cuộc trò chuyện vào session_state
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    # Nếu là cuộc trò chuyện mới, tạo ID mới
    if not conv_id:
        conv_id = str(uuid.uuid4())

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    st.session_state.conversations[conv_id] = {
        "title": title,
        "messages": messages.copy(),
        "timestamp": timestamp
    }

    return conv_id


# Upload PDF
def upload_pdf():
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose PDF file to upload", type=["pdf"])
    if uploaded_file:
        st.success(f"{uploaded_file.name} uploaded successfully")
        # st.text(f"Size: {round(uploaded_file.size / 1024 / 1024, 2)} MB")
    return uploaded_file


# Lỗi
def conversation_history():
    if st.button("Start new chat"):
        st.session_state.messages = []
        st.session_state.current_conversation = None
        st.rerun
    st.subheader("Recent")
    for conv_id, conv_data in st.session_state.conversations.items():
        if st.button(f"{conv_data['title']}", key=f"conv_{conv_id}"):
            st.session_state.messages = conv_data["messages"].copy()
            st.session_state.current_conversation = conv_id
            st.rerun


def main():
    st.title("Boring Chat")

    # Khởi tạo session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "chats" not in st.session_state:
        st.session_state.chats = {}

    # Khởi tạo mô hình
    model = configure_gemini_api()

    # Tạo sidebar với hai phần
    with st.sidebar:
        # Phần 1: Lịch sử trò chuyện
        # st.header("Lịch sử trò chuyện")

        # Nút tạo cuộc trò chuyện mới
        # conversation_history()
        if st.button("Start new chat"):
            st.session_state.messages = []
            st.session_state.current_conversation = None
            st.rerun()

        # Hiển thị các cuộc trò chuyện cũ
        st.subheader("Recent")
        for conv_id, conv_data in st.session_state.conversations.items():
            if st.button(f"{conv_data['title']}", key=f"conv_{conv_id}"):
                st.session_state.messages = conv_data["messages"].copy()
                st.session_state.current_conversation = conv_id
                st.rerun()

        # Phần 2: Upload PDF
        st.markdown("---")
        uploaded_file = upload_pdf()
        # st.header("Upload PDF")
        # uploaded_file = st.file_uploader("Choose PDF file to upload", type=["pdf"])
        #
        # if uploaded_file:
        #     st.success(f"{uploaded_file.name} uploaded successfully")
        #     st.text(f"Size: {round(uploaded_file.size / 1024 / 1024, 2)} MB")

    # Hiển thị tin nhắn chat
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])

    # Lấy đối tượng chat cho cuộc trò chuyện hiện tại
    chat = multi_conversation(model, st.session_state.current_conversation)

    # Khu vực chat input
    prompt = st.chat_input("Enter your message...")

    if prompt:
        # Thêm tin nhắn người dùng vào chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)

        with st.spinner("Loading..."):
            # Tạo phản hồi sử dụng đối tượng chat để duy trì ngữ cảnh
            response = generate_response(chat, prompt, uploaded_file)

            # Thêm phản hồi trợ lý vào messages
            st.session_state.messages.append({"role": "assistant", "content": response})
            display_message("assistant", response)

            # Lưu cuộc trò chuyện
            if not st.session_state.current_conversation:
                # Sử dụng prompt đầu tiên làm tiêu đề (cắt ngắn nếu quá dài)
                title = prompt
                if len(title) > 30:
                    title = title[:27] + "..."

                # Lưu cuộc trò chuyện và lưu chat object
                conv_id = save_conversation(title, st.session_state.messages)
                st.session_state.current_conversation = conv_id
                st.session_state.chats[conv_id] = chat
            else:
                # Cập nhật cuộc trò chuyện hiện tại
                save_conversation(
                    st.session_state.conversations[st.session_state.current_conversation]["title"],
                    st.session_state.messages,
                    st.session_state.current_conversation
                )


if __name__ == "__main__":
    main()
