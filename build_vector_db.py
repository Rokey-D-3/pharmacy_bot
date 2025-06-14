#!/usr/bin/env python3

# pharmacy_bot/build_vector_db.py
import os
from dotenv import load_dotenv

# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    # pharmacy_bot/ 디렉터리 기준
    base_path = os.path.dirname(os.path.abspath(__file__))
    resource_path = os.path.join(base_path, "resource")

    # .env 로드
    load_dotenv(dotenv_path=os.path.join(base_path, ".env"))
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print(".env에 OPENAI_API_KEY가 없습니다.")
        return

    # 텍스트 파일 경로
    txt_path = os.path.join(resource_path, "drug_symptom_mapping.txt")
    if not os.path.exists(txt_path):
        print(f"파일이 존재하지 않습니다: {txt_path}")
        return

    # 텍스트 불러오기
    loader = TextLoader(txt_path)
    documents = loader.load()

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embedding 설정
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 벡터 DB 생성 및 저장
    db_path = os.path.join(resource_path, "chroma_db")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    db.persist()

    print(f"Vector DB 생성 완료 → {db_path}")

if __name__ == "__main__":
    main()
