import sys
import os
from langchain.chains import RetrievalQAWithSourcesChain
# from langchain_openai import ChatOpenAI
from common import (
    load_api_keys,
    initialize_pinecone,
    initialize_embeddings,
    initialize_vector_store,
    initialize_prompt,
    initialize_llm
)

def main():
    
    # 1) Load API keys
    OPENAI_API_KEY, PINECONE_API_KEY, LANGSMITH_API_KEY = load_api_keys()
    print(LANGSMITH_API_KEY)

    # 2) Initialize Pinecone client
    index = initialize_pinecone(PINECONE_API_KEY)

    # 3) Initialize OpenAI embeddings
    embedding_model = initialize_embeddings(OPENAI_API_KEY)
    
    # 4) Create Pinecone vector store
    vector_store = initialize_vector_store(index, embedding_model)

    # Connect to Langsmith
    os.environ['LANGSMITH_TRACING'] ='true'
    os.environ['LANGSMITH_ENDPOINT']="https://api.smith.langchain.com"
    os.environ['LANGSMITH_PROJECT']="berlin-city-agent"

    # 5) Configure the ChatOpenAI model
    # Specify the model you want to use for RetrievalQA
    llm_model = "gpt-4o-mini"  
    llm = initialize_llm(OPENAI_API_KEY, model=llm_model)

    # 6) Configure RetrievalQAWithSourcesChain with the combine_documents_chain
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    # 7) Query from user
    query = "Wie kann ich einen Reisepass beantragen?"

    # 8) Get the response from RetrievalQAWithSourcesChain
    response = qa_chain.invoke(query)

    # 9) Display the response
    print("Response:", response)

if __name__ == "__main__":
    main()