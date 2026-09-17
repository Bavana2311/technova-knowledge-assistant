import os
import yaml
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_embedding_model(provider: str, config: dict):

    if provider == "openai":
        return OpenAIEmbeddings(
            model=config["embedding"]["openai_model"]
        )

    elif provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=config["embedding"]["gemini_model"]
        )

    else:
        raise ValueError("Provider must be 'openai' or 'gemini'")


def get_index_path(provider: str, config: dict):

    if provider == "openai":
        return config["vectordb"]["faiss_openai"]

    elif provider == "gemini":
        return config["vectordb"]["faiss_gemini"]

    else:
        raise ValueError("Provider must be 'openai' or 'gemini'")


def create_retriever(chunks, provider: str, config: dict):

    index_path = get_index_path(provider, config)

    # Convert relative path to absolute path
    index_path = os.path.abspath(index_path)

    embedding_model = get_embedding_model(provider, config)

    os.makedirs(index_path, exist_ok=True)

    vectorstore = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vectorstore.save_local(index_path)

    print(f"✅ Created FAISS index ({provider}) at:")
    print(index_path)

    return vectorstore


def load_retriever(provider: str, config: dict):

    index_path = get_index_path(provider, config)

    # Convert relative path to absolute path
    index_path = os.path.abspath(index_path)

    embedding_model = get_embedding_model(provider, config)

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at: {index_path}"
        )

    vectorstore = FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print(f"✅ Loaded FAISS index ({provider}) from:")
    print(index_path)

    return vectorstore


def get_retriever(config: dict, chunks_if_needed=None):

    provider = config["llm"]["provider"]

    index_path = get_index_path(provider, config)

    index_path = os.path.abspath(index_path)

    if not os.path.exists(
        os.path.join(index_path, "index.faiss")
    ):
        if chunks_if_needed is None:
            raise RuntimeError(
                "Chunks must be provided when creating a new index."
            )

        return create_retriever(
            chunks_if_needed,
            provider,
            config
        )

    return load_retriever(
        provider,
        config
    )