from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from common import (
    load_api_keys,
    initialize_pinecone,
    initialize_embeddings,
    initialize_vector_store,
    initialize_prompt,
    initialize_llm
)
from langchain.chains import RetrievalQA

def initialize_agent_system():
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
    # Specify a different model if desired, e.g., "gpt-4"
    llm_model = "gpt-4"  # Choose based on your subscription and needs
    llm = initialize_llm(OPENAI_API_KEY, model=llm_model)

    # 7) Configure Retrieval QA chain with the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": custom_prompt}
    )

    # 8) Initialize conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,  # Number of past interactions to retain
        return_messages=True
    )

    # 9) Convert RetrievalQA chain into a Tool
    knowledge_tool = Tool(
        name='Knowledge Base',
        func=qa_chain.run,
        description=(
            'Use this tool to answer questions about Berlin services by retrieving relevant information from the knowledge base.'
        )
    )

    # 10) Initialize the agent with the tool and conversational memory
    agent = initialize_agent(
        tools=[knowledge_tool],
        llm=llm,
        agent="chat-conversational-react-description",  # Agent type suitable for conversational tasks
        verbose=True,
        max_iterations=3,  # Maximum number of iterations the agent can perform to answer a query
        early_stopping_method='generate',  # Method to stop the agent early if needed
        memory=conversational_memory
    )

    return agent

def chat_with_agent(agent):
    print("Willkommen beim Berlin Service Assistenten! (Tippe 'exit' zum Beenden)")
    while True:
        user_input = input("Du: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Beenden der Konversation. Auf Wiedersehen!")
            break
        try:
            response = agent.run(user_input)
            print(f"Assistent: {response}\n")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}\n")

def main():
    agent = initialize_agent_system()
    chat_with_agent(agent)

if __name__ == "__main__":
    main()