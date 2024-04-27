import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import  SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

api_key_folder = "keys"

# Read the API key from the file
with open(os.path.join(api_key_folder, "gemini.txt"), "r") as f:
    api_key = f.read().strip()

# Load the embedding model using the API key read from the file
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")


# Load the embedding model
# embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyDeFmoBbE6wNDKftGcF0mowbgzhC5HjzUw",model="models/embedding-001")

# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Define retrieval function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Define chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""YAs your dedicated AI Assistant, 
    my goal is to offer swift and effective support tailored to your needs. 
    Whether you require answers, guidance, or information, I'm equipped to provide assistance efficiently."""),
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context.\nContext:\n{context}\nQuestion:\n{question}\nAnswer:")
])

# Initialize chat model
chat_model = ChatGoogleGenerativeAI(google_api_key="api_key",
                                    model="gemini-1.5-pro-latest")

# Initialize output parser
output_parser = StrOutputParser()

# Define RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template  | chat_model  | output_parser
)

# Streamlit UI
st.title("🌍An Advanced RAG System Based on 'Leave No Context Behind'")

question = st.text_input("Enter Your question: ⁉️")

if st.button("📝Loading Answer"):
    if question:
        response = rag_chain.invoke(question)
        st.write(response)
    else:
        st.warning("📑Please enter a context based on paper.")
