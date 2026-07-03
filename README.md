# VectorMind-RAG: Production-Grade Enterprise Retrieval Engine



An engineering-focused, in-memory Retrieval-Augmented Generation (RAG) conversational platform built from scratch using Python, Google Gemini, and NumPy. 



Instead of relying on high-level orchestration abstractions (like LangChain), this project implements the mathematical core of semantic search—including custom text chunking pipelines, native vector embeddings via `text-embedding-004`, and raw matrix distance operations using Cosine Similarity.



\## 🚀 Key Features



\* \*\*Custom Token-Distance Chunking:\*\* Implements an algorithmic text-splitting pipeline with sliding-window chunk overlaps to ensure no semantic context is lost across borders.

\* \*\*In-Memory Vector Space:\*\* Utilizes `st.session\_state` caching mechanics to simulate an ultra-low latency vector database, ensuring documents are only embedded once per session to control API overhead costs.

\* \*\*Mathematical Semantic Alignment:\*\* Uses raw NumPy matrix operations to compute high-dimensional Cosine Similarity scores ($1024$-dimensions) to rank relevant context segments against a user's question.

\* \*\*Anti-Hallucination Guardrails:\*\* Employs strict corporate context boundary prompts to prevent LLM hallucinations, forcing the backend engine to fail gracefully if facts aren't present in the source files.



\## 🛠️ The Tech Stack



\* \*\*Language:\*\* Python

\* \*\*UI Engine:\*\* Streamlit (Wide Layout Responsive Dashboard)

\* \*\*Mathematical Operations:\*\* NumPy

\* \*\*GenAI Ecosystem:\*\* Google GenAI SDK (`gemini-2.5-flash`, `text-embedding-004`)

\* \*\*Environment Security:\*\* Python-Dotenv



\---



\## 📐 System Architecture \& Data Flow



1\. \*\*Document Ingestion:\*\* The user uploads a `.txt` knowledge base through the Streamlit sidebar.

2\. \*\*Preprocessing:\*\* The raw text is broken into overlapping 600-character tokens.

3\. \*\*Vector Vectorization:\*\* Every chunk is sent to `text-embedding-004`, converting language into multi-dimensional arrays.

4\. \*\*Query Matching:\*\* The user's typed chat question is converted into a retrieval query vector.

5\. \*\*Matrix Ranking:\*\* Cosine similarity measures the inner product of vectors to pull the Top-2 most semantically matching text blocks.

6\. \*\*Grounded Synthesis:\*\* The text blocks are injected into an evaluation prompt layout and processed safely by `gemini-2.5-flash`.



\---



\## ⚙️ Installation \& Local Setup



\### 1. Clone the Repository

```bash

git clone \[https://github.com/YOUR\_USERNAME/your-repo-name.git](https://github.com/YOUR\_USERNAME/your-repo-name.git)

cd your-repo-name

