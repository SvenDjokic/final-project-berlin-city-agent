import sys
from common import (
    load_api_keys,
    initialize_pinecone,
    initialize_embeddings,
    initialize_vector_store,
    initialize_prompt,
    initialize_llm
)
from langchain.chains import RetrievalQA

def main():
    # 1) Load API keys
    OPENAI_API_KEY, PINECONE_API_KEY = load_api_keys()

    # 2) Initialize Pinecone client
    index = initialize_pinecone(PINECONE_API_KEY)

    # 3) Initialize OpenAI embeddings
    embedding_model = initialize_embeddings(OPENAI_API_KEY)

    # 4) Create Pinecone vector store
    vector_store = initialize_vector_store(index, embedding_model)

    # 5) Initialize custom prompt
    custom_prompt = initialize_prompt()

    # 6) Configure the ChatOpenAI model
    # Specify the model you want to use for RetrievalQA
    llm_model = "gpt-4o-mini"  
    llm = initialize_llm(OPENAI_API_KEY, model=llm_model)

    # 7) Configure Retrieval QA chain with the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": custom_prompt}
    )

    # 8) Query from user
    query = "Wie kann ich einen Reisepass beantragen?"

    # 9) Get the response from RetrievalQA
    response = qa_chain.run(query)

    # 10) Display the response
    print("Response:", response)

if __name__ == "__main__":
    main()