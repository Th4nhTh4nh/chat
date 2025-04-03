import re
import google.generativeai as genai
import os
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv
from chromadb import EmbeddingFunction


load_dotenv()


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def split_text(text):
    splited_text = re.split('\n \n', text)
    return [i for i in splited_text if i != ""]


def create_chroma_db(documents, path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    for id, doc in enumerate(documents):
        db.add(documents=doc, ids=str(id))
    return db, name


def load_chroma_collection(path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    return db


def get_relevant_passage(query, db, n_results):
    passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    return passage


def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace("'", "").replace("\n", " ")
    prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
        strike a friendly and converstional tone. \
        If the passage is irrelevant to the answer, you may ignore it.
        QUESTION: '{query}'
        PASSAGE: '{relevant_passage}'

    ANSWER:
        """).format(query=query, relevant_passage=escaped)

    return prompt


def generate_response(prompt):
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("Missing API key")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(os.getenv("MODEL_NAME"))
    response = model.generate_content(prompt)
    return response.text


def generate_answer(db, query):
    relevant_text = get_relevant_passage(query, db, n_results=3)
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))
    answer = generate_response(prompt)
    return answer


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        gemini_api_key = os.getenv("API_KEY")
        if not gemini_api_key:
            raise ValueError("Missing API key")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Test query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]





def main():
    file_path = "Spiderum - Tại sao địa lý nước Nga TỆ.pdf"
    question = "Vấn đề địa lý mà nước Nga phải đối mặt là gì?"
    pdf_text = load_pdf(
        path=file_path)
    chunked_text = split_text(text=pdf_text)
    db, name = create_chroma_db(documents=chunked_text, path="E:\pythonProject\RAG\contents", name="rag_experiment")
    db = load_chroma_collection(path="E:\pythonProject\RAG\contents", name="rag_experiment")
    answer = generate_answer(db, query=question)
    print(answer)

if __name__ == "__main__":
    main()
