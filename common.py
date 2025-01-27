import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain.prompts import PromptTemplate

def load_api_keys():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY or PINECONE_API_KEY in environment variables.")
    return OPENAI_API_KEY, PINECONE_API_KEY

def initialize_pinecone(PINECONE_API_KEY, index_name="berlin-services-retrieval-agent"):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in [info["name"] for info in pc.list_indexes()]:
        raise ValueError(f"Pinecone index '{index_name}' does not exist.")
    index = pc.Index(index_name)
    return index

def initialize_embeddings(OPENAI_API_KEY, model="text-embedding-ada-002"):
    embedding_model = OpenAIEmbeddings(
        model=model,
        api_key=OPENAI_API_KEY
    )
    return embedding_model

def initialize_vector_store(index, embedding_model, text_key="text_excerpt"):
    vector_store = LangChainPinecone(
        index=index,
        embedding=embedding_model,
        text_key=text_key
    )
    return vector_store

def create_custom_prompt():
    custom_prompt = PromptTemplate(
        template="""
        You are an expert assistant specializing in providing detailed and comprehensive information about Berlin services.

        Use the following pieces of context to answer the question at the end thoroughly. Ensure that your response includes step-by-step instructions, all relevant details, and any necessary explanations. Present the information in a clear, organized format, such as a numbered list or bullet points. If the information is not contained within the context, reply that you don't have the information.

        Example:
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

        Context:
        {context}

        Question: {question}
        Answer:""",
        input_variables=["context", "question"]
    )
    return custom_prompt

def initialize_llm(OPENAI_API_KEY, model, temperature=0.0):
    llm = ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=temperature
    )
    return llm

def initialize_prompt():
    return create_custom_prompt()