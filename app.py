import streamlit as st
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API Key
load_dotenv()

st.set_page_config(page_title="VectorMind RAG Chatbot", page_icon="📚", layout="wide")

# App Layout Titles
st.title("📚 VectorMind RAG Chatbot")
st.caption("Powered by Gemini Embeddings, In-Memory Vector Search, and Gemini 2.5 Flash")

# Initialize Gemini Client
client = genai.Client()

# Helper Function: Split document into chunks
def chunk_text(text, chunk_size=600, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# Helper Function: Generate embeddings using Gemini
def get_embedding(text_to_embed, is_query=False):
    task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    response = client.models.embed_content(
        model="gemini-embedding-001",  # <-- Changed this string
        contents=text_to_embed,
        config=types.EmbedContentConfig(task_type=task)
    )
    # Extract the vector list from the response object
    return response.embeddings[0].values

# Helper Function: Cosine Similarity calculation
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2) if (norm_v1 and norm_v2) else 0.0

# --- SIDEBAR: Document Upload & Knowledge Base Processing ---
with st.sidebar:
    st.header("📁 Knowledge Base Setup")
    uploaded_file = st.file_uploader("Upload a text document (.txt)", type=["txt"])
    
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8")
        
        # Avoid re-processing if the file hasn't changed
        if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
            with st.spinner("Processing document & generating vector embeddings..."):
                chunks = chunk_text(raw_text)
                embeddings = [get_embedding(chunk, is_query=False) for chunk in chunks]
                
                # Store in session state (Our localized Vector Database)
                st.session_state.kb_chunks = chunks
                st.session_state.kb_vectors = embeddings
                st.session_state.file_name = uploaded_file.name
            st.success(f"Successfully indexed {len(chunks)} text chunks!")
    else:
        st.info("Please upload a document to enable the context-aware RAG pipeline.")

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your document..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Default context if no file is uploaded
    context = "No document provided by the user."
    
    # RAG Pipeline Logic: Retrieve relevant chunks if Knowledge Base exists
    if "kb_vectors" in st.session_state and len(st.session_state.kb_vectors) > 0:
        with st.spinner("Searching vector space for answers..."):
            query_vector = get_embedding(prompt, is_query=True)
            
            # Compute similarity scores between query and all stored document chunks
            scores = [cosine_similarity(query_vector, doc_vec) for doc_vec in st.session_state.kb_vectors]
            
            # Extract the top 2 most matching text chunks
            top_indices = np.argsort(scores)[-2:][::-1]
            retrieved_chunks = [st.session_state.kb_chunks[idx] for idx in top_indices]
            context = "\n\n".join(retrieved_chunks)

    # System instruction template forcing the LLM to ground its answers
    system_prompt = f"""
    You are an expert, precision-focused enterprise RAG assistant. Your job is to answer the user's question using ONLY the provided context snippets below. 
    If the answer cannot be confidently derived from the context, state exactly: "I cannot find the answer within the provided document." Do not hallucinate or use external knowledge.

    CONTEXT SNIPPETS:
    {context}
    """

    # Generate response from Gemini
    with st.chat_message("assistant"):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[system_prompt, f"User Question: {prompt}"]
            )
            answer = response.text
        except Exception as e:
            answer = f"Error generating AI response: {e}"
            
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})