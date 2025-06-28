import os
import logging
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


DATA_PATH = os.getcwd() + "/data/"
DB_FAISS_PATH = os.getcwd() + "/vectorstores/"

# Function to load vector store
def get_vector_store():
    logger.info("Loading FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    try:
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS vector store loaded successfully.")
        st.success("FAISS vector store loaded successfully.")
    except KeyError as e:
        logger.error(f"Failed to load FAISS vector store: {e}")
        st.error(f"Failed to load FAISS vector store: {e}")
    return vectorstore

# Function to get the retriever chain for document search
def get_retriever_chain(vector_store):
    logger.info("Setting up retriever chain...")
    llm = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", convert_system_message_to_human=True)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    logger.info("Retriever chain setup complete.")
    return history_retriever_chain

# Function to get the response from the retriever chain
def get_response(user_input):
    logger.info("Generating response for user input...")
    history_retriever_chain = get_retriever_chain(st.session_state.vector_store)
    llm = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", convert_system_message_to_human=True)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("user", "Answer the user's questions based on the below context and suggest follow-up questions:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)

    # Create final retrieval chain
    conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)
    response = conversational_retrieval_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    logger.info(f"Response generated: {response['answer']}")
    return response["answer"]

# Ingestion code to process PDFs and create the FAISS vector store
def create_vector_db_from_pdf(pdf_file):
    logger.info("Processing PDF file to create FAISS vector store...")
    # Save the uploaded PDF file
    pdf_path = os.path.join(DATA_PATH, pdf_file.name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    logger.info(f"PDF saved at {pdf_path}")
    
    # Load and split the documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info("PDF loaded and split into documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents=documents)
    logger.info(f"Documents split into {len(texts)} chunks.")

    # Create FAISS embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    logger.info("FAISS embeddings created.")

    # Save the FAISS vector store
    db.save_local(DB_FAISS_PATH)
    logger.info(f"Vector store saved to {DB_FAISS_PATH}")
    st.success("Vector store created and saved successfully!")

# Streamlit app
if __name__ == '__main__':
    logger.info("Starting Streamlit app...")

    st.title("PDF-based Q&A Chatbot")

    # Sidebar to upload PDF
    pdf_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

    # Ingest PDF button
    if st.sidebar.button("Ingest PDF and Generate FAISS Index"):
        logger.info("Ingest PDF button clicked.")
        if pdf_file is not None:
            create_vector_db_from_pdf(pdf_file)
        else:
            st.error("Please upload a PDF file first.")
            logger.warning("No PDF file uploaded.")

    # Load FAISS vector store
    if os.path.exists(DB_FAISS_PATH):
        st.session_state.vector_store = get_vector_store()
    else:
        st.warning("No FAISS vector store found. Please upload a PDF and ingest it first.")
        logger.warning("No FAISS vector store found. Prompting user to upload a PDF.")

    # Initialize chat history if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            HumanMessage(content="Hi"),
            AIMessage(content="I am a bot, how can I help you with your document?")
        ]
        logger.info("Initialized chat history.")

    # Handle user input and generate response
    user_input = st.chat_input("Type your message here...")
    if user_input is not None and user_input.strip() != "":
        logger.info(f"User input received: {user_input}")
        response = get_response(user_input)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        else:
            with st.chat_message("Human"):
                st.write(message.content)

    logger.info("Streamlit app is running and waiting for user input.")
