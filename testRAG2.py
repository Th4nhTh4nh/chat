from PyPDF2 import PdfReader
from typing import List, Any, Tuple
import re
from collections import deque
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models import Collection
import os
import uuid
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

def text_extract(pdf_path: str) -> str:
    pdf_pages = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            pdf_pages.append(text)
    pdf_text = "\n".join(pdf_pages)
    return pdf_text


def text_chunk(text: str, max_length: int = 1000) -> List[str]:
    sentences = deque(re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')))
    chunks = []
    chunk_text = ""
    while sentences:
        sentence = sentences.popleft().strip()
        if sentence:
            if len(chunk_text) + len(sentence) > max_length and chunk_text:
                chunks.append(chunk_text)
                chunk_text = sentence
            else:
                chunk_text += " " + sentence
    if chunk_text:
        chunks.append(chunk_text)
    return chunks


def create_vector_store(db_path: str) -> Collection:
    client = chromadb.PersistentClient(path=db_path)
    embeddings = embedding_functions.DefaultEmbeddingFunction()
    # embeddings = embedding_functions.OpenAIEmbeddingFunction(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model_name="text-embedding-3-small"
    # )
    db = client.create_collection(
        name="pdf_chunks",
        embedding_function=embeddings
    )
    return db


def insert_chunks_vectordb(chunks: List[str], db: Collection, file_path: str) -> None:
    file_name = os.path.basename(file_path)
    id_list = [str(uuid.uuid4()) for _ in range(len(chunks))]
    metadata_list = [{"chunk": i, "source": file_name} for i in range(len(chunks))]
    batch_size = 40
    for i in range(0, len(chunks), batch_size):
        end_id = min(i + batch_size, len(chunks))
        db.add(
            documents=chunks[i:end_id],
            metadatas=metadata_list[i:end_id],
            ids=id_list[i:end_id]
        )
    print(f"{len(chunks)} chunks added to the vector store")


def retrieve_chunks(db: Collection, query: str, n_results: int = 2) -> List[Any]:
    # Thực hiện truy vấn trên cơ sở dữ liệu để có được các phần có liên quan nhất
    relevant_chunks = db.query(query_texts=[query], n_results=n_results)
    return relevant_chunks


def build_context(relevant_chunks) -> str:
    # kết hợp văn bản từ các chunk có liên quan với dấu phân cách dòng mới
    context = "\n".join(relevant_chunks['documents'][0])
    return context


def get_context(pdf_path: str, query: str, db_path: str) -> Tuple[str, str]:
    if os.path.exists(db_path):
        print("Loading existing vector store...")

        client = chromadb.PersistentClient(path=db_path)

        # Khoi tao hamf Embedding
        embeddings = embedding_functions.DefaultEmbeddingFunction()

        # Dua pdf chunk vao vector store
        db = client.get_collection(name="pdf_chunks", embedding_function=embeddings)
    else:
        print("Creating new vector store...")

        # Trich xuat text tu file pdf
        pdf_text = text_extract(pdf_path)

        # Chunk the extracted text
        chunks = text_chunk(pdf_text)

        # Tạo một vector store mới
        db = create_vector_store(db_path)

        # Load các chunk văn bản vào kho lưu trữ vector
        insert_chunks_vectordb(chunks, db, pdf_path)

    # Lấy các khối chunk có liên quan dựa trên truy vấn
    relevant_chunks = retrieve_chunks(db, query)

    # Xây dựng bối cảnh (context) từ các chunk có liên quan
    context = build_context(relevant_chunks)

    # trả về context và truy vấn
    return context, query


def get_prompt(context: str, query: str) -> str:
    # Định dạng prompt với context và query được cung cấp
    rag_prompt = f""" You are an AI model trained for question answering. You should answer the
    given question based on the given context combine with your own knowledge.
    Question : {query}
    \n
    Context : {context}
    \n
    If the answer is not present in the given context, respond as: The answer to this question is not available
    in the provided content.
    """

    return rag_prompt


def get_response(rag_prompt: str) -> str:
    model = "gemini/gemini-2.0-flash"

    messages = [{"role": "user", "content": rag_prompt}]

    response = completion(model=model, messages=messages, temperature=0)

    answer = response.choices[0].message.content
    return answer


def rag_pipeline(pdf_path: str, query: str, db_path: str) -> str:
    context, query = get_context(pdf_path, query, db_path)

    rag_prompt = get_prompt(context, query)

    response = get_response(rag_prompt)

    return response


def main():
    # chroma DB path
    current_dir = "content/rag"
    persistent_directory = os.path.join(current_dir, "db", "chroma_db_pdf")

    pdf_path = "Spiderum - Tại sao địa lý nước Nga TỆ.pdf"

    # RAG query
    query1 = "Hãy trả lời 1 cách cụ thể và đưa ra dẫn chứng cho câu hỏi Vấn đề của địa lý nước Nga là gì?"
    query = "Hành động của NATO là gì và nó ảnh hưởng thế nào tới Nga?"
    # RAG pipeline
    answer = rag_pipeline(pdf_path, query, persistent_directory)

    print(f"Query:{query}")
    print(f"Generated answer:{answer}")


if __name__ == "__main__":
    main()
