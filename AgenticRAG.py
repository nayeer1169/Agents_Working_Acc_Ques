import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from azure_llm_loader import load_llm
from embed import embeddings
import json

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

def get_vectorstore(text_chunks):
    return FAISS.from_texts(text_chunks, embeddings)

def get_conversational_chain(vectorstore):
    llm = load_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain

def agent1_extract_intent_entities(question, llm):
    print("\n▶ Agent 1: Extracting intent and entities...")
    
    prompt = f"""
You are an intent/entity extraction model.
Extract the intent and key entities from the question below.
Return ONLY a valid JSON in this exact format:
{{"intent": "intent_description", "entities": ["entity1", "entity2", "..."]}}

Question: {question}
"""
    response = llm.predict(prompt).strip()

    try:
        parsed = json.loads(response)
        print(" Agent 1 Output:", parsed)
        return parsed
    except json.JSONDecodeError:
        print(" Agent 1: Failed to parse response:\n", response)
        return {"intent": "", "entities": []}


def agent2_refine_intent_entities(intent_entities, llm):
    print("\n▶ Agent 2: Refining extracted intent and entities...")
    
    prompt = f"""
You are validating and enhancing extracted intent/entities.

Input JSON:
{json.dumps(intent_entities)}

Ensure this structure:
{{"intent": "intent_string", "entities": ["entity1", "entity2", "..."]}}

Return a corrected version in the same format (no nested objects, all strings).
"""
    response = llm.predict(prompt).strip()

    try:
        parsed = json.loads(response)
        print(" Agent 2 Output:", parsed)
        return parsed
    except json.JSONDecodeError:
        print(" Agent 2: Failed to parse response:\n", response)
        return intent_entities

def agent3_retrieve_chunks(intent_entities, vectorstore, k=3):
    print("\n▶ Agent 3: Retrieving top chunks using intent/entities...")
    query = f"{intent_entities['intent']} {' '.join(intent_entities['entities'])}"
    print(f"Agent 3 Query: {query}")
    docs = vectorstore.similarity_search(query, k=k)
    chunks = "\n\n".join([doc.page_content for doc in docs])
    print(f" Agent 3 Retrieved {len(docs)} chunks.")
    return chunks

def agent4_generate_answer(question, context_chunks, llm):
    print("\n▶ Agent 4: Generating final answer...")
    prompt = f"You are answering this question: '{question}'\nUsing the following document context:\n{context_chunks}\n\nAnswer:"
    answer = llm.predict(prompt).strip()
    print(f" Agent 4 Answer:\n{answer}")
    return answer



def main():
    st.set_page_config("Chat with PDF", layout="wide")
    st.title("Chat with PDF Files - Multi-Agent Version")

    if "chain" not in st.session_state:
        st.session_state.chain = None

    with st.sidebar:
        st.header("Upload your PDFs")
        pdf_docs = st.file_uploader("Select PDFs", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Reading and indexing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.chain = get_conversational_chain(vectorstore)
            st.success("Ready to chat!")

    if st.session_state.chain:
        user_question = st.text_input("Ask a question about your PDFs:")
        if user_question:
            llm = load_llm()

            # Agent 1: Extract intent and entities
            intent_entities = agent1_extract_intent_entities(user_question, llm)

            # Agent 2: Refine the extracted data
            refined_intent_entities = agent2_refine_intent_entities(intent_entities, llm)

            # Agent 3: Retrieve relevant chunks
            context_chunks = agent3_retrieve_chunks(refined_intent_entities, st.session_state.vectorstore)

            # Agent 4: Generate the final response
            final_response = agent4_generate_answer(user_question, context_chunks, llm)

            st.write("Answer:", final_response)

if __name__ == "__main__":
    main()
