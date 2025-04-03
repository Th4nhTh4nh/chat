import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import os
import uuid
import re
import chromadb
from chromadb import EmbeddingFunction
from pypdf import PdfReader
from docx import Document
import pandas as pd
from dotenv import load_dotenv
import tempfile
import time
from datetime import datetime
import hashlib
import logging
from sqlite import *

# Thiết lập logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Cấu hình và hằng số
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_NUM_RESULTS = 3
TEMPERATURE = 0.2


# RAG
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        try:
            gemini_api_key = os.getenv("API_KEY")
            if not gemini_api_key:
                raise ValueError("Missing API key")
            genai.configure(api_key=gemini_api_key)
            model = "models/embedding-001"
            title = "Document Embedding"

            results = []
            for text in input:
                try:
                    embedding = genai.embed_content(
                        model=model,
                        content=text,
                        task_type="retrieval_document",
                        title=title
                    )["embedding"]
                    results.append(embedding)
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error embedding text: {str(e)}")
                    results.append([0.0] * 768)
            return results
        except Exception as e:
            logger.error(f"Embedding function error: {str(e)}")
            return [[0.0] * 768 for _ in range(len(input))]


# Xử lý nhiều loại tài liệu
def extract_text_from_file(file_path, file_type):
    try:
        if file_type == 'pdf':
            return extract_text_from_pdf(file_path)
        elif file_type == 'docx':
            return extract_text_from_docx(file_path)
        elif file_type == 'txt':
            return extract_text_from_txt(file_path)
        elif file_type == 'csv':
            return extract_text_from_csv(file_path)
        else:
            return f"Unsupported file type: {file_type}"
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return f"Error processing file: {str(e)}"


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    return text


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n\n"
    return text


def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)


# Phương pháp chunking
def split_text_into_chunks(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split text into overlapping chunks of specified size."""
    chunks = []

    # Phân chia đoạn văn cơ bản đầu tiên
    paragraphs = re.split(r'\n\s*\n', text)
    current_chunk = ""

    for paragraph in paragraphs:
        # Bỏ qua văn bản trống
        if not paragraph.strip():
            continue

        # Nếu việc thêm đoạn văn này vượt quá kích thước chunk, lưu chunk hiện tại và bắt đầu một chunk mới
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Giữ lại một số phần chunk chồng chéo từ phần chunk trước
            overlap_size = min(chunk_overlap, len(current_chunk))
            current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else ""

        # Thêm đoạn văn vào chunk hiện tại
        current_chunk += paragraph + "\n\n"

    # Thêm chunk còn lại cuối cùng nếu có
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Nếu chúng ta không có chunk nào (ví dụ: tài liệu rất ngắn), hãy trả về toàn bộ văn bản dưới dạng một chunk
    if not chunks:
        chunks = [text.strip()]

    # Thêm metadata vào chunks (e.g., vị trí trong tài liệu)
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        # Thêm index dưới dạng metadata
        chunk_with_metadata = {
            "text": chunk,
            "metadata": {
                "chunk_id": i,
                "position": f"{i}/{len(chunks)}",
                "char_count": len(chunk)
            }
        }
        enhanced_chunks.append(chunk_with_metadata)

    return enhanced_chunks


def create_chroma_collection(document_chunks, collection_name):
    """Create a ChromaDB collection from document chunks with metadata."""
    chroma_dir = "./chroma_db"
    os.makedirs(chroma_dir, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    # Kiểm tra xem có tồn tại không, nếu có, hãy xóa nó để tạo lại
    try:
        existing_collection = chroma_client.get_collection(name=collection_name)
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    # Tạo mới
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=GeminiEmbeddingFunction()
    )

    # Trích xuất văn bản và metadata từ chunk
    texts = [chunk["text"] for chunk in document_chunks]
    metadatas = [chunk["metadata"] for chunk in document_chunks]
    ids = [f"chunk_{i}" for i in range(len(document_chunks))]

    # Thêm tài liệu theo từng đợt để tránh các giới hạn
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        try:
            collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        except Exception as e:
            logger.error(f"Error adding batch to collection: {str(e)}")

    return collection, collection_name


def load_chroma_collection(collection_name):
    chroma_dir = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    return chroma_client.get_collection(
        name=collection_name,
        embedding_function=GeminiEmbeddingFunction()
    )


def get_relevant_passages(query, collection, n_results=DEFAULT_NUM_RESULTS):
    """Truy xuất các đoạn văn có liên quan và metadata của chúng."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Kết hợp các tài liệu với metadata của chúng để có ngữ cảnh tốt hơn
        passages = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            relevance = 1.0 - min(results['distances'][0][i] / 2.0, 0.99)  # Chuẩn hóa khoảng cách đến điểm liên quan

            passages.append({
                "text": doc,
                "metadata": metadata,
                "relevance": f"{relevance:.2f}"
            })

        return passages
    except Exception as e:
        logger.error(f"Error retrieving passages: {str(e)}")
        return []


def make_rag_prompt(query, relevant_passages):
    """Tạo prompt gồm các đoạn văn có liên quan và metadata của chúng."""
    # Định dạng đoạn văn với metadata của chúng
    formatted_passages = []

    for i, passage in enumerate(relevant_passages):
        formatted_passage = f"[Passage {i + 1} - Relevance: {passage['relevance']}]\n{passage['text']}"
        formatted_passages.append(formatted_passage)

    context = "\n\n".join(formatted_passages)

    prompt = """You are a helpful and informative assistant that answers questions based on the provided document passages. 
    Follow these guidelines:

    1. Answer using information from the provided passages
    2. If the answer isn't contained in the passages, say "The document doesn't contain information about that."
    3. Cite specific parts of the passages to support your answer
    4. Respond in a conversational and helpful tone
    5. Break down complex concepts for non-technical users
    6. Keep your answer concise but comprehensive

    QUESTION: {query}

    RELEVANT PASSAGES:
    {context}

    ANSWER:
    """.format(query=query, context=context)

    return prompt


# Cấu hình Gemini API
def configure_gemini_api():
    genai.configure(api_key=os.getenv("API_KEY"))
    model = genai.GenerativeModel(
        os.getenv("MODEL_NAME"),
        generation_config={"temperature": TEMPERATURE}
    )
    return model


# Ghi nhớ ngữ cảnh hội thoại
def multi_conversation(model, conv_id=None):
    if conv_id is None or conv_id not in st.session_state.chats:
        chat = model.start_chat(history=[])
        if conv_id:
            st.session_state.chats[conv_id] = chat
        return chat
    return st.session_state.chats[conv_id]


# Tạo câu trả lời kết hợp RAG
def generate_response(chat, prompt, file=None, use_rag=False, history=None):
    try:
        if history and len(history) > 0:
            recent_history = history[-5:]
            for msg in recent_history:
                if msg["role"] == "user":
                    chat.send_message(msg["content"])
        start_time = time.time()

        if use_rag and "current_collection" in st.session_state:
            # Dùng RAG đ tạo phản hồi
            collection = st.session_state.current_collection

            # Lấy các thông tin liên quan
            relevant_passages = get_relevant_passages(
                prompt,
                collection,
                n_results=st.session_state.get("num_results", DEFAULT_NUM_RESULTS)
            )

            # Lưu các đoạn văn đã lấy vào session state để hiển thị
            st.session_state.last_retrieved_passages = relevant_passages

            # Tạo prompt RAG
            rag_prompt = make_rag_prompt(prompt, relevant_passages)

            # Gen ra phản hồi
            response = chat.send_message(rag_prompt)

            # Thời gian xử lý
            processing_time = time.time() - start_time
            logger.info(f"RAG response generated in {processing_time:.2f} seconds")

            return response.text

        elif file:
            # Xử lý tệp không dùng RAG bằng phương pháp ban đầu
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file.name.split(".")[-1]}') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name

            with open(tmp_file_path, 'rb') as f:
                file_content = f.read()

            # Đường dẫn file
            file_part = {"mime_type": f"application/{file.name.split('.')[-1]}", "data": file_content}

            # Gửi tin nhắn với cả văn bản và file
            response = chat.send_message([prompt, file_part])

            # Xóa file lưu tạm thời
            os.unlink(tmp_file_path)

            return response.text

        else:
            response = chat.send_message(prompt)
            return response.text

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"I encountered an error while generating a response: {str(e)}"


# def display_message(role, content):
#     with st.chat_message(role):
#         st.write(content)

def display_message(role, content):
    if role == "user":
        st.markdown(
            f"<div style='background-color: #E3F2FD; padding: 10px; border-radius: 8px; margin: 5px 0;'>{content}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color: #F5F5F5; padding: 10px; border-radius: 8px; margin: 5px 0;'>{content}</div>",
            unsafe_allow_html=True
        )


def type_writer_effect(text, speed=0.003):
    container = st.empty()
    displayed_text = ""

    for char in text:
        displayed_text += char
        container.markdown(displayed_text)
        time.sleep(speed)

    return container


# Lưu trữ tài liệu
def save_document_metadata(filename, file_type, collection_name=None):
    """Lưu metadata về tài liệu đã tải lên."""
    if "documents" not in st.session_state:
        st.session_state.documents = {}

    doc_id = hashlib.md5(f"{filename}_{datetime.now()}".encode()).hexdigest()

    st.session_state.documents[doc_id] = {
        "filename": filename,
        "type": file_type,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "collection_name": collection_name,
    }

    return doc_id


# Lưu hội thoại
def save_conversation(title, messages, conv_id=None):
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    if not conv_id:
        conv_id = str(uuid.uuid4())

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    st.session_state.conversations[conv_id] = {
        "title": title,
        "messages": messages.copy(),
        "timestamp": timestamp,
        "rag_enabled": st.session_state.get("rag_enabled", False),
        "collection_name": st.session_state.get("collection_name", None),
        "document_id": st.session_state.get("current_document_id", None),
        "rag_settings": {
            "chunk_size": st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
            "chunk_overlap": st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            "num_results": st.session_state.get("num_results", DEFAULT_NUM_RESULTS),
        }
    }

    return conv_id


# Upload và xử lý tài liệu
def upload_and_process_document():
    st.header("Upload Document")

    supported_types = ["pdf", "docx", "txt", "csv"]
    uploaded_file = st.file_uploader(
        "Upload a document to chat with",
        type=supported_types
    )

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()
        st.success(f"{uploaded_file.name} uploaded successfully")

        # Hiển thị cài đặt RAG nếu có file được tải lên
        with st.expander("RAG Settings"):
            col1, col2, col3 = st.columns(3)
            with col1:
                chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=100,
                    max_value=2000,
                    value=st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE)
                )
            with col2:
                chunk_overlap = st.number_input(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=500,
                    value=st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
                )
            with col3:
                num_results = st.number_input(
                    "Results to Retrieve",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.get("num_results", DEFAULT_NUM_RESULTS)
                )

            # Lưu cài đặt vào session state
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.num_results = num_results

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Process with RAG"):
                with st.spinner("Processing document... This may take a minute"):
                    # Lưu tệp tải lên như 1 file tạm thời
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Truy xuất văn bản từ tài liệu
                    document_text = extract_text_from_file(tmp_file_path, file_type)

                    # Xóa file tạm
                    os.unlink(tmp_file_path)

                    if not document_text or len(document_text) < 10:
                        st.error("Could not extract text from document. Please try another file.")
                        return uploaded_file

                    # Chia nhỏ văn bản thành chunk với metadata
                    text_chunks = split_text_into_chunks(
                        document_text,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap
                    )

                    # Tạo tên collection duy nhất
                    collection_name = f"{file_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

                    # Tạo collection
                    collection, _ = create_chroma_collection(text_chunks, collection_name)

                    # Lưu metadata của tài liệu
                    doc_id = save_document_metadata(uploaded_file.name, file_type, collection_name)

                    # Lưu trữ vào session state
                    st.session_state.current_collection = collection
                    st.session_state.collection_name = collection_name
                    st.session_state.rag_enabled = True
                    st.session_state.current_document_id = doc_id

                    st.success(f"Document processed with {len(text_chunks)} chunks. Ready for questions!")

        with col2:
            if st.button("Use Standard Mode"):
                st.session_state.rag_enabled = False
                st.session_state.pop("current_collection", None)
                st.session_state.pop("collection_name", None)
                st.success("Using standard mode with document")

    return uploaded_file


# Hiển thị phân tích tài liệu
def document_analysis():
    if "last_retrieved_passages" in st.session_state and st.session_state.rag_enabled:
        st.subheader("Retrieved Passages")

        passages = st.session_state.last_retrieved_passages

        for i, passage in enumerate(passages):
            with st.expander(f"Passage {i + 1} - Relevance: {passage['relevance']}"):
                st.markdown(passage["text"])
                st.caption(f"Position: {passage['metadata'].get('position', 'Unknown')}")


# Quản lý tài liệu
def document_management():
    if "documents" in st.session_state and st.session_state.documents:
        st.subheader("Document Library")

        for doc_id, doc_data in st.session_state.documents.items():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(f"📄 {doc_data['filename']}")

            with col2:
                st.caption(f"Uploaded: {doc_data['upload_time']}")

            with col3:
                if doc_data.get("collection_name") and st.button("Use", key=f"use_{doc_id}"):
                    try:
                        collection = load_chroma_collection(doc_data["collection_name"])
                        st.session_state.current_collection = collection
                        st.session_state.collection_name = doc_data["collection_name"]
                        st.session_state.current_document_id = doc_id
                        st.session_state.rag_enabled = True
                        st.success(f"Now using {doc_data['filename']} for RAG")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not load document: {str(e)}")


def main():
    st.set_page_config(
        page_title="Lỏser CHat",
        page_icon="📚",
        # layout="wide"
    )

    with open("static/styles.css", mode="r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.markdown("<h1 style='font-size: 32px; color: #333;'>Lỏe CHat</h1>", unsafe_allow_html=True)
    # Khởi tạo session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
    if "num_results" not in st.session_state:
        st.session_state.num_results = DEFAULT_NUM_RESULTS

    # Khởi tạo mô hình
    model = configure_gemini_api()

    # Layout với sidebar và khu vực chính
    # col_sidebar, col_main = st.columns([1, 3])

    # Sidebar
    # with col_sidebar:
    with st.sidebar:
        # Trạng thái RAG
        if st.session_state.rag_enabled:
            st.success("📚 RAG Mode: Enabled")

            if "current_document_id" in st.session_state and st.session_state.current_document_id:
                doc_data = st.session_state.documents.get(st.session_state.current_document_id, {})
                if doc_data:
                    st.info(f"Using document: {doc_data.get('filename', 'Unknown')}")
        else:
            st.info("RAG Mode: Disabled")

        # Nút tạo hội thoại mới
        if st.button("New Conversation"):
            st.session_state.messages = []
            st.session_state.current_conversation = None
            st.rerun()

        # Tab cho lịch sử và quản lý tài liệu
        tabs = st.tabs(["Conversations", "Documents", "Settings"])
        with tabs[0]:
            # Hiển thị các cuộc trò chuyện cũ
            if st.session_state.conversations:
                for conv_id, conv_data in st.session_state.conversations.items():
                    # Hiển thị thông báo RAG cho các cuộc trò chuyện với RAG
                    rag_indicator = "🔍 " if conv_data.get("rag_enabled", False) else ""
                    if st.button(f"{rag_indicator}{conv_data['title']}", key=f"conv_{conv_id}"):
                        st.session_state.messages = conv_data["messages"].copy()
                        st.session_state.current_conversation = conv_id

                        # Khôi phục cài đặt RAG nếu có thể
                        if conv_data.get("rag_enabled", False) and conv_data.get("collection_name"):
                            try:
                                st.session_state.current_collection = load_chroma_collection(
                                    conv_data["collection_name"])
                                st.session_state.collection_name = conv_data["collection_name"]
                                st.session_state.rag_enabled = True

                                # Khôi phục cài đặt RAG
                                rag_settings = conv_data.get("rag_settings", {})
                                st.session_state.chunk_size = rag_settings.get("chunk_size", DEFAULT_CHUNK_SIZE)
                                st.session_state.chunk_overlap = rag_settings.get("chunk_overlap",
                                                                                  DEFAULT_CHUNK_OVERLAP)
                                st.session_state.num_results = rag_settings.get("num_results", DEFAULT_NUM_RESULTS)

                                # Khôi phục tài liệu tham khảo
                                st.session_state.current_document_id = conv_data.get("document_id")
                            except Exception as e:
                                st.error(f"Could not load RAG data: {str(e)}")
                                st.session_state.rag_enabled = False
                        else:
                            st.session_state.rag_enabled = False

                        st.rerun()
            else:
                st.write("No conversations yet.")

        with tabs[1]:
            # Quản lý tài liệu
            document_management()

        with tabs[2]:
            # Cài đặt
            st.subheader("AI Settings")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=TEMPERATURE,
                step=0.1,
                help="Higher values make output more random, lower values more deterministic"
            )

    uploaded_file = upload_and_process_document()

    # Giao diện chat
    st.markdown("<h2 style='font-size: 24px; color: #555; margin-top: 20px;'>Chat</h2>", unsafe_allow_html=True)

    # Hiển thị lịch sủ chat
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])

    # Hiển thị phân tích tài liệu
    document_analysis()

    # Khu vực nhập tin nhắn mới
    user_input = st.chat_input("Enter your message...")

    if user_input:
        # Thêm tin nhắn của người dùng vào lịch sử chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message("user", user_input)

        # Nhận chat hiện tại hoặc tạo chat mới
        current_chat = multi_conversation(model, st.session_state.current_conversation)

        with st.spinner("Thinking..."):
            # Gen ra phản hồi
            if uploaded_file and not st.session_state.rag_enabled:
                # Xử lý file mà ko dùng RAG
                response = generate_response(
                    current_chat,
                    user_input,
                    file=uploaded_file,
                    history=st.session_state.messages)
            else:
                # Xử lý bằng RAG nếu được bật hoặc chỉ văn bản nếu không
                response = generate_response(
                    current_chat,
                    user_input,
                    use_rag=st.session_state.rag_enabled,
                    history=st.session_state.messages
                )

            # Thêm phản hồi vào mục lịch sử
            st.session_state.messages.append({"role": "assistant", "content": response})
            display_message("assistant", response)

            # HIệu ứng khi phản hồi
            # st.session_state.messages.append({"role": "assistant", "content": response})
            # with st.chat_message("assistant"):
            #     type_writer_effect(response)

            # Lưu hội thoại
            if len(st.session_state.messages) >= 2:
                # Tạo tiêu đề từ tin nhắn đầu tiên của chat
                first_message = next((msg for msg in st.session_state.messages if msg["role"] == "user"), None)
                if first_message:
                    title = first_message["content"][:30] + "..." if len(first_message["content"]) > 30 else \
                        first_message["content"]
                else:
                    title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

                # Lưu
                conv_id = save_conversation(title, st.session_state.messages, st.session_state.current_conversation)
                st.session_state.current_conversation = conv_id

    # Khu vực hiển thị content
    # with col_main:
    # Tải file lên
    # uploaded_file = upload_and_process_document()
    #
    # # Giao diện chat
    # st.markdown("<h2 style='font-size: 24px; color: #555; margin-top: 20px;'>Chat</h2>", unsafe_allow_html=True)
    #
    # # Hiển thị lịch sủ chat
    # for message in st.session_state.messages:
    #     display_message(message["role"], message["content"])
    #
    # # Hiển thị phân tích tài liệu
    # document_analysis()
    #
    # # Khu vực nhập tin nhắn mới
    # user_input = st.chat_input("Enter your message...")
    #
    # if user_input:
    #     # Thêm tin nhắn của người dùng vào lịch sử chat
    #     st.session_state.messages.append({"role": "user", "content": user_input})
    #     display_message("user", user_input)
    #
    #     # Nhận chat hiện tại hoặc tạo chat mới
    #     current_chat = multi_conversation(model, st.session_state.current_conversation)
    #
    #     with st.spinner("Thinking..."):
    #         # Gen ra phản hồi
    #         if uploaded_file and not st.session_state.rag_enabled:
    #             # Xử lý file mà ko dùng RAG
    #             response = generate_response(
    #                 current_chat,
    #                 user_input,
    #                 file=uploaded_file,
    #                 history=st.session_state.messages)
    #         else:
    #             # Xử lý bằng RAG nếu được bật hoặc chỉ văn bản nếu không
    #             response = generate_response(
    #                 current_chat,
    #                 user_input,
    #                 use_rag=st.session_state.rag_enabled,
    #                 history=st.session_state.messages
    #             )
    #
    #         # Thêm phản hồi vào mục lịch sử
    #         st.session_state.messages.append({"role": "assistant", "content": response})
    #         display_message("assistant", response)
    #
    #         # st.session_state.messages.append({"role": "assistant", "content": response})
    #         # with st.chat_message("assistant"):
    #         #     type_writer_effect(response)
    #
    #         # Lưu hội thoại
    #         if len(st.session_state.messages) >= 2:
    #             # Tạo tiêu đề từ tin nhắn đầu tiên của chat
    #             first_message = next((msg for msg in st.session_state.messages if msg["role"] == "user"), None)
    #             if first_message:
    #                 title = first_message["content"][:30] + "..." if len(first_message["content"]) > 30 else \
    #                     first_message["content"]
    #             else:
    #                 title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    #
    #             # Lưu
    #             conv_id = save_conversation(title, st.session_state.messages, st.session_state.current_conversation)
    #             st.session_state.current_conversation = conv_id


if __name__ == "__main__":
    main()
