import os
from dotenv import load_dotenv
import time
import pandas as pd
from langchain_openai import OpenAIEmbeddings                # NEW
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        yield " ".join(chunk_words)
        i += chunk_size - overlap

# Path to dataset
file_path = "berlin_services/output.json"

# Load DataFrame
data = pd.read_json(file_path)
print("Number of pages:", len(data))
print(data.head(2))

# Prepare OpenAI embeddings
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

# Pinecone setup
api_key = os.getenv("PINECONE_API_KEY") 
pc = Pinecone(api_key=api_key)
spec = ServerlessSpec(cloud="aws", region="us-east-1")

index_name = "berlin-services-retrieval-agent"
existing_indexes = [info["name"] for info in pc.list_indexes()]

# Create index if needed
if index_name not in existing_indexes:
    pc.create_index(index_name, dimension=1536, metric='dotproduct', spec=spec)
    # wait for index to be fully ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)
time.sleep(1)

# Let's define a batch size for how many vectors we upsert at once
batch_size = 100
vector_buffer = []

for page_idx in tqdm(range(len(data)), desc="Processing pages"):
    record = data.iloc[page_idx]

    # Extract text, metadata
    page_text = record["text"]
    page_title = record["title"]
    page_url = record["url"]
    page_sub_links = record.get("sub_links", [])

    # 1) Chunk the text
    chunks = list(chunk_text(page_text, chunk_size=400, overlap=50))

    # 2) Embed each chunk
    chunk_embeddings = embed.embed_documents(chunks)

    # 3) Build vector data for each chunk
    for chunk_i, (chunk_str, chunk_emb) in enumerate(zip(chunks, chunk_embeddings)):
        # We'll create a unique ID for each chunk
        vector_id = f"doc_{page_idx}_chunk_{chunk_i}"

        # Create metadata
        meta = {
            "title": page_title,
            "url": page_url,
            "sub_links": page_sub_links,
            "chunk_i": chunk_i,
            "text_excerpt": chunk_str[:200]  # optional partial excerpt
        }

        # Add to our buffer
        vector_buffer.append((vector_id, chunk_emb, meta))

    # 4) Periodically upsert
    if len(vector_buffer) >= batch_size:
        index.upsert(vectors=vector_buffer)
        vector_buffer = []

# If anything remains in the buffer, upsert it at the end
if vector_buffer:
    index.upsert(vectors=vector_buffer)

stats = index.describe_index_stats()
print("Index stats:", stats)
