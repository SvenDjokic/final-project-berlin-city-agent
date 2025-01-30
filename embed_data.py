import time
import pandas as pd
from tqdm.auto import tqdm
# import openai  # Needed for specific exception handling
from common import (
    load_api_keys,
    initialize_pinecone,
    initialize_embeddings,
    initialize_vector_store
)

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        yield " ".join(chunk_words)
        i += chunk_size - overlap

def retry_function(func, max_retries=3, delay=5, *args, **kwargs):
    """
    Attempts to execute a function with retries.

    :param func: The function to execute.
    :param max_retries: Maximum number of retry attempts.
    :param delay: Delay (in seconds) between retries.
    :param args: Positional arguments for the function.
    :param kwargs: Keyword arguments for the function.
    :return: The result of the function if successful.
    :raises: The last exception encountered if all retries fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[DEBUG] Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                print(f"[DEBUG] Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"[DEBUG] All {max_retries} attempts failed.")
                raise

def main():
    # 1) Load API keys
    OPENAI_API_KEY, PINECONE_API_KEY, LANGSMITH_API_KEY = load_api_keys()

    # 2) Initialize Pinecone client
    index = initialize_pinecone(PINECONE_API_KEY)

    # 3) Initialize OpenAI embeddings
    embedding_model = initialize_embeddings(OPENAI_API_KEY)

    # 4) Create Pinecone vector store
    vector_store = initialize_vector_store(index, embedding_model)

    # 5) Path to dataset
    file_path = "berlin_services/output.json"

    # 6) Load DataFrame
    data = pd.read_json(file_path)
    print("Number of pages:", len(data))

    batch_size = 100
    vector_buffer = []

    # Loop through all pages
    for page_idx in tqdm(range(len(data)), desc="Processing pages"):
        record = data.iloc[page_idx]

        page_text = record["text"]
        page_title = record["title"]
        page_url = record["url"]

        # DEBUG PRINT 1: Starting chunk
        print(f"[DEBUG] Starting to chunk page {page_idx} with title '{page_title}'...")

        chunks = list(chunk_text(page_text, chunk_size=1000, overlap=50))
        # DEBUG PRINT 2: After chunk
        print(f"[DEBUG] Finished chunking page {page_idx}. Number of chunks: {len(chunks)}")

        # 2) Embedding with retry
        print(f"[DEBUG] Now embedding page {page_idx} with {len(chunks)} chunks...")
        try:
            chunk_embeddings = retry_function(
                embedding_model.embed_documents,
                max_retries=3,
                delay=2,
                texts=chunks
            )
            print(f"[DEBUG] Successfully embedded page {page_idx}.")
        except Exception as e:
            print(f"[ERROR] Failed to embed page {page_idx} after retries. Skipping this page.")
            continue  # Skip to the next page

        # DEBUG PRINT 3: After embedding
        print(f"[DEBUG] Finished embedding page {page_idx}. Sleeping 12s...")
        time.sleep(12)  # Increased sleep time after embedding

        # 3) Build vector data for each chunk
        for chunk_i, (chunk_str, chunk_emb) in enumerate(zip(chunks, chunk_embeddings)):
            vector_id = f"doc_{page_idx}_chunk_{chunk_i}"
            meta = {
                "title": page_title,
                "url": page_url,
                "source": page_url,
                "chunk_i": chunk_i,
                "text_excerpt": chunk_str
            }
            vector_buffer.append((vector_id, chunk_emb, meta))

        # DEBUG PRINT 4: buffer size
        print(f"[DEBUG] vector_buffer size is {len(vector_buffer)} after page {page_idx}.")

        # Upsert if buffer is large enough
        if len(vector_buffer) >= batch_size:
            print(f"[DEBUG] Upserting {len(vector_buffer)} vectors to Pinecone...")
            try:
                retry_function(
                    index.upsert,
                    max_retries=3,
                    delay=2,
                    vectors=vector_buffer
                )
                print(f"[DEBUG] Successfully upserted {len(vector_buffer)} vectors.")
            except Exception as e:
                print(f"[ERROR] Failed to upsert vectors for page {page_idx} after retries.")
            # after upsert
            print(f"[DEBUG] Done upserting. Sleeping 12s...")
            time.sleep(12)  # Increased sleep time after upserting
            vector_buffer = []

    # Final leftover upsert
    if vector_buffer:
        print(f"[DEBUG] Final upsert of leftover {len(vector_buffer)} vectors...")
        try:
            retry_function(
                index.upsert,
                max_retries=3,
                delay=2,
                vectors=vector_buffer
            )
            print(f"[DEBUG] Successfully upserted final {len(vector_buffer)} vectors.")
        except Exception as e:
            print(f"[ERROR] Failed to upsert final vectors after retries.")
        print(f"[DEBUG] Done final upsert. Sleeping 12s...")
        time.sleep(12)  # Increased sleep time after final upsert

    stats = index.describe_index_stats()
    print("Index stats:", stats)

if __name__ == "__main__":
    main()
