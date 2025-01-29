import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def load_api_keys():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY or PINECONE_API_KEY in environment variables.")
    return OPENAI_API_KEY, PINECONE_API_KEY, LANGSMITH_API_KEY

def initialize_pinecone(PINECONE_API_KEY, index_name="berlin-services-retrieval"):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [info["name"] for info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"[DEBUG] Pinecone index '{index_name}' does not exist. Creating a new one...")
        # Define the index configuration
        dimension = 1536  # Ensure this matches the embedding dimension of your model (e.g., text-embedding-ada-002)
        metric = 'dotproduct'  

        # Define the specification for the index
        spec = ServerlessSpec(cloud="aws", region="us-east-1")

        # Create the index
        pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)

        # Wait until the index is ready
        index_ready = False
        while not index_ready:
            try:
                status = pc.describe_index(index_name).status['ready']
                if status:
                    index_ready = True
                    print(f"[DEBUG] Pinecone index '{index_name}' is ready.")
                else:
                    print("[DEBUG] Waiting for Pinecone index to be ready...")
            except Exception as e:
                print(f"[DEBUG] Error checking index status: {e}")
            if not index_ready:
                import time
                time.sleep(2)  # Wait for 2 seconds before retrying
    
    else:
        print(f"[DEBUG] Pinecone index '{index_name}' already exists.")
    
    index = pc.Index(index_name)
    return index

def initialize_embeddings(OPENAI_API_KEY, model="text-embedding-ada-002"):
    embedding_model = OpenAIEmbeddings(
        model=model,
        api_key=OPENAI_API_KEY
    )
    return embedding_model

def initialize_vector_store(index, embedding_model, text_key="text_excerpt"):
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        text_key=text_key,
        # include_metadata=True 
    )
    return vector_store

def initialize_llm(OPENAI_API_KEY, model, temperature=0.0):
    llm = ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=temperature
    )
    return llm

def initialize_prompt():
    return create_custom_prompt()