import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def main():
    # 1) Load API keys from environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        print("[ERROR] Missing OPENAI_API_KEY or PINECONE_API_KEY in environment variables.")
        return

    # 2) Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "berlin-services-retrieval-agent"

    if index_name not in [info["name"] for info in pc.list_indexes()]:
        print(f"[ERROR] Pinecone index '{index_name}' does not exist.")
        return
    index = pc.Index(index_name)

    # 3) Initialize OpenAI embeddings
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY
    )

    # 4) Create Pinecone vector store
    vector_store = LangChainPinecone(
        index=index,
        embedding=embedding_model,
        text_key="text_excerpt"  # Ensure this matches your embed_data.py
    )

    # 5) Convert vector store to retriever
    retriever = vector_store.as_retriever()

    # 6) Configure the ChatOpenAI model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # Use a valid model name
        api_key=OPENAI_API_KEY,
        temperature=0.0  # For deterministic responses
    )

    # 7) Define a custom prompt
    custom_prompt = PromptTemplate(
        template="""
        You are an expert assistant specializing in providing detailed and comprehensive information about Berlin services.

        Use the following pieces of context to answer the question at the end thoroughly. Ensure that your response includes step-by-step instructions, all relevant details, and any necessary explanations. Present the information in a clear, organized format, such as a numbered list or bullet points. Use bold headings for each step. If the information is not contained within the context, reply that you don't have the information.

        Example 1:
        Context:
        In Berlin, die Anmeldung erfolgt beim örtlichen Bürgeramt. Sie müssen einen gültigen Personalausweis, ein aktuelles biometrisches Passfoto und einen Nachweis über Ihre Wohnadresse vorlegen. Die Bearbeitungszeit beträgt in der Regel eine Woche, und die Gebühren variieren je nach Stadtteil.

        Question: "Wie melde ich mich in Berlin an?"
        Answer:
        1. **Terminvereinbarung:** Vereinbaren Sie einen Termin beim örtlichen Bürgeramt in Berlin. Dies kann oft online oder telefonisch erfolgen.
        2. **Erforderliche Dokumente:** Bringen Sie folgende Unterlagen mit:
        - Einen gültigen Personalausweis
        - Ein aktuelles biometrisches Passfoto
        - Ihren Mietvertrag oder Nachweis der Wohnadresse
        3. **Gebühren:** Zahlen Sie die Anmeldegebühren, die je nach Stadtteil variieren können.
        4. **Bearbeitungszeit:** Die Bearbeitung dauert in der Regel etwa eine Woche.
        5. **Abschluss:** Nach der Bearbeitung erhalten Sie eine Meldebescheinigung.

        Example 2:
        Context:
        Um einen Führerschein in Berlin zu erwerben, müssen Sie eine Fahrschule besuchen, eine theoretische und praktische Prüfung ablegen und die entsprechenden Gebühren bezahlen. Die gesamte Ausbildung dauert in der Regel mehrere Monate, abhängig von Ihrem Lernfortschritt und den Verfügbarkeit der Prüfungen.

        Question: "Wie bekomme ich einen Führerschein in Berlin?"
        Answer:
        1. **Fahrschule auswählen:** Wählen Sie eine Fahrschule in Ihrer Nähe und melden Sie sich dort an.
        2. **Theoretischer Unterricht:** Nehmen Sie am theoretischen Unterricht teil und bereiten Sie sich auf die theoretische Prüfung vor.
        3. **Praktischer Unterricht:** Absolvieren Sie die praktischen Fahrstunden mit einem Fahrlehrer.
        4. **Prüfungen ablegen:** Legen Sie die theoretische und praktische Prüfung ab.
        5. **Gebühren bezahlen:** Zahlen Sie die anfallenden Gebühren für die Fahrschule und die Prüfungen.
        6. **Führerschein erhalten:** Nach bestandenen Prüfungen erhalten Sie Ihren Führerschein.

        Context:
        {context}

        Question: {question}
        Answer:""",
        input_variables=["context", "question"]
    )


    # 8) Configure Retrieval QA chain with the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    # 9) Query from user
    query = "Wie kann ich einen Reisepass beantragen?"

    # 10) Get the response from RetrievalQA
    response = qa_chain.invoke(query)

    # 11) Display the response
    print("Response:", response)

if __name__ == "__main__":
    main()