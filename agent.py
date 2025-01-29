import os
import json
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.prompts import SystemMessagePromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.agents import initialize_agent
from common import (
    load_api_keys,
    initialize_pinecone,
    initialize_embeddings,
    initialize_vector_store,
    initialize_llm
)

CUSTOM_PROMPT = """
You are an expert assistant specializing in providing detailed and comprehensive information about Berlin services.
Answer queries step by step, ensuring clarity and accuracy. Always include sources at the end if relevant.

Example:
User: "Wie melde ich mich in Berlin an?"

Assistant:
1. **Terminvereinbarung:** Vereinbaren Sie einen Termin beim örtlichen Bürgeramt in Berlin. Dies kann oft online oder telefonisch erfolgen.
2. **Erforderliche Dokumente:** Bringen Sie folgende Unterlagen mit:
   - Einen gültigen Personalausweis
   - Ein aktuelles biometrisches Passfoto
   - Ihren Mietvertrag oder Nachweis der Wohnadresse
3. **Gebühren:** Die Gebühren variieren je nach Stadtteil.
4. **Bearbeitungszeit:** Die Bearbeitung dauert in der Regel etwa eine Woche.
5. **Abschluss:** Nach der Bearbeitung erhalten Sie eine Meldebescheinigung.

Now, respond to the following question:

User: {input}

Assistant:
"""

def initialize_qa_chain_with_sources(llm, retriever):
    """
    Returns a RetrievalQAWithSourcesChain that will also return source_documents.
    Useful for debugging or for building custom tools that need URLs.
    """
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return chain

def initialize_agent_system():
    """
    This function handles:
    1. Loading API Keys
    2. Initializing Pinecone + vector store
    3. Initializing LLM
    4. Building a QA chain with sources
    5. Creating Tools for the agent
    6. Creating the agent with a system prompt
    7. Setting up conversation memory
    Returns the agent object.
    """
    # 1) Load API keys
    OPENAI_API_KEY, PINECONE_API_KEY, LANGSMITH_API_KEY = load_api_keys()

    # 2) Initialize Pinecone client & vector store
    index = initialize_pinecone(PINECONE_API_KEY)
    embedding_model = initialize_embeddings(OPENAI_API_KEY)
    vector_store = initialize_vector_store(index, embedding_model)
    retriever = vector_store.as_retriever()

    # 3) Connect to Langsmith
    os.environ['LANGSMITH_TRACING'] ='true'
    os.environ['LANGSMITH_ENDPOINT']="https://api.smith.langchain.com"
    os.environ['LANGSMITH_PROJECT']="berlin-city-agent"

    # 4) Initialize the LLM
    llm_model = "gpt-4o-mini"  
    llm = initialize_llm(OPENAI_API_KEY, model=llm_model, temperature=0.0)

    # 5) Build a QA chain that returns sources
    qa_chain_with_sources = initialize_qa_chain_with_sources(llm, retriever)

    # 6) Define Tools
    def search_berlin_services_tool(user_input: str) -> str:
        """
        This tool retrieves the answer + source docs from the chain,
        then formats them into a single response string that includes the source URLs.
        """
        result = qa_chain_with_sources({"question": user_input})
        answer = result["answer"]
        source_docs = result["source_documents"]

        # Collect unique source URLs from metadata
        urls = []
        for doc in source_docs:
            meta = doc.metadata
            url = meta.get("url") or meta.get("source") or "N/A"
            if url not in urls:
                urls.append(url)

        # Format the sources to be appended at the end
        if urls:
            sources_str = "\n\n**Quellen/URLs**:\n" + "\n".join(f"- {u}" for u in urls)
        else:
            sources_str = "\n\n(Keine spezifischen Quellen gefunden.)"

        return answer + sources_str

    def summarize_berlin_info_tool(user_input: str) -> str:
        """
        Similar to search_berlin_services_tool, but could be used
        if the user explicitly wants a short summary. We still show sources for transparency.
        """
        result = qa_chain_with_sources({"question": user_input})
        answer = result["answer"]
        source_docs = result["source_documents"]

        # Gather sources
        urls = []
        for doc in source_docs:
            meta = doc.metadata
            url = meta.get("url") or meta.get("source") or "N/A"
            if url not in urls:
                urls.append(url)

        # Make them visible in the response
        if urls:
            sources_str = "\n\n**Quellen/URLs**:\n" + "\n".join(f"- {u}" for u in urls)
        else:
            sources_str = "\n\n(Keine spezifischen Quellen gefunden.)"

        # Possibly shorten the answer or keep it as is
        return "Kurze Zusammenfassung:\n" + answer + sources_str
    
    tools = [
        Tool(
            name="SearchBerlinServices",
            func=search_berlin_services_tool,
            description=(
                "Use this tool to retrieve detailed info about Berlin services. "
                "It will provide step-by-step answers with sources."
            )
        ),
        Tool(
            name="SummarizeBerlinInfo",
            func=summarize_berlin_info_tool,
            description=(
                "Use this tool to provide a concise summary about Berlin services. "
                "Always includes relevant URLs."
            )
        )
    ]

    # 6) Define the System Prompt using SystemMessagePromptTemplate
    system_prompt = SystemMessagePromptTemplate.from_template(CUSTOM_PROMPT)


    # 7) Configure conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,  # Number of past interactions to retain
        return_messages=True
    )

    def retrieve_documents_with_metadata(query: str) -> str:
        docs = vector_store.as_retriever().get_relevant_documents(query)
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return json.dumps(results, ensure_ascii=False)

    # 10) Initialize the agent 
    agent = initialize_agent(
        tools=tools,  
        llm=llm,  
        agent=AgentType.OPENAI_FUNCTIONS, 
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        handle_parsing_errors=True,
        memory=conversational_memory,
        agent_kwargs={
            "system_message": system_prompt,
            "memory_key": "chat_history"
        }
    )

    return agent

def chat_with_agent(agent):
    '''
    Simple loop to chat with the agent in the terminal.
    '''
    print("Willkommen beim Berlin Service Assistenten! (Tippe 'exit' zum Beenden)")
    while True:
        user_input = input("Du: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Beenden der Konversation. Auf Wiedersehen!")
            break

        try:
            # AgentType.OPENAI_FUNCTIONS expects a dict: {"input": "..."}
            response = agent.invoke({"input": user_input})
            print("\nAssistent:", response, "\n")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}\n")

def main():
    agent = initialize_agent_system()
    chat_with_agent(agent)

if __name__ == "__main__":
    main()