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
Answer queries step by step, ensuring clarity and accuracy. 

Important: The final answer must only include the links that are strictly relevant to the user's query. Omit any link that is NOT relevant to the user's question. Include links that ARE relevant.

Respond to the user's question but do NOT generate references or links yourself.
We (the system) will append any relevant source links automatically after your summary.

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
        Summarize the answer from the retrieved chunks,
        then append the real metadata 'url' and 'title'
        from those chunks.
        """
        # 1) Get the retrieval result (with sources)
        result = qa_chain_with_sources({"question": user_input})
        summary = result["answer"]  # Summarized text from the LLM
        source_docs = result["source_documents"]  # The chunks used

        # Debug print
        print(f"[DEBUG] Retrieved {len(source_docs)} docs.")
        for i, doc in enumerate(source_docs):
            print(f"[DEBUG] Doc {i} metadata: {doc.metadata}")

        # 2) Build a list of unique sources
        used_sources = []
        for doc in source_docs:
            meta = doc.metadata
            # Pull the real URL from metadata
            url = meta.get("url", "N/A")
            title = meta.get("title", "N/A")
        
            # Ensure no duplicates
            if (url, title) not in used_sources:
                used_sources.append((url, title))

        # 3) Convert the sources to a neat string
        if used_sources:
            sources_str = "\n\n**Weiterführende Links**:\n"
            for url, title in used_sources:
                # Format as a Markdown link or "Title (URL)"
                sources_str += f"- [{title}]({url})\n"
        else:
            sources_str = "\n\n(Keine spezifischen Quellen gefunden.)"
        
        # 4) Combine the answer text with the correct source links
        final_answer = summary + sources_str
        
        return final_answer

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
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
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