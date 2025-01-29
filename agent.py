from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQAWithSourcesChain
from common import (
    load_api_keys,
    initialize_pinecone,
    initialize_embeddings,
    initialize_vector_store,
    initialize_prompt,
    initialize_llm
)

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
    # Specify llm model
    llm_model = "gpt-4o-mini"  
    llm = initialize_llm(OPENAI_API_KEY, model=llm_model)

    # 7) Configure RetrievalQAWithSourcesChain with the custom prompt
    qa_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
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
        func=qa_chain.invoke,
        description=(
            'Use this tool to answer questions about Berlin services by retrieving relevant information from the knowledge base. '
            'Each answer will include the source URLs where the information was retrieved from.'
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
        memory=conversational_memory,
        custom_prompt=custom_prompt
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
            response = agent.invoke(user_input)
            print(f"Assistent: {response}\n")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}\n")

def main():
    agent = initialize_agent_system()
    chat_with_agent(agent)

if __name__ == "__main__":
    main()