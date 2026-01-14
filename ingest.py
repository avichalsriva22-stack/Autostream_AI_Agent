import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def ingest() -> None:
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    project_root = Path(__file__).resolve().parent
    source_path = project_root / "knowledge.md"

    if not source_path.exists():
        raise FileNotFoundError(f"Missing knowledge base at {source_path}")

    loader = TextLoader(str(source_path), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vector_store = _build_vector_store(chunks)

    store_dir = project_root / "vectorstore"
    store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(store_dir))


def _build_vector_store(chunks):
    try:
        logging.info("Building FAISS store with OpenAI embeddings")
        embeddings = OpenAIEmbeddings()
        return FAISS.from_documents(chunks, embeddings)
    except Exception as openai_error:  # pylint: disable=broad-except
        logging.warning(
            "OpenAI embeddings failed (%s). Falling back to sentence-transformers",
            openai_error,
        )
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            return FAISS.from_documents(chunks, embeddings)
        except Exception as huggingface_error:  # pylint: disable=broad-except
            raise RuntimeError(
                "Failed to build vector store with available embedding backends"
            ) from huggingface_error


if __name__ == "__main__":
    ingest()
