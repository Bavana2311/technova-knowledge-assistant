# TechNova Knowledge Assistant

1. Project Overview

TechNova Knowledge Assistant is an Enterprise Knowledge Assistant built using Advanced Retrieval-Augmented Generation (RAG).

The system allows employees to ask questions about company policies and documents. It retrieves relevant information from internal documents and generates grounded answers using Google Gemini.

The system is designed to reduce hallucinations by answering only from the retrieved document context.

---

2. Key Features

- Multi-format document ingestion
- PDF, DOCX and TXT support
- Document chunking
- Gemini embeddings
- FAISS vector database
- BM25 keyword-based retrieval
- Hybrid retrieval
- Cross-encoder reranking
- Conversational memory
- Context-aware follow-up questions
- Hallucination handling
- Gemini LLM-based answer generation
- Streamlit user interface

---

3. System Architecture

Documents
    ↓
Document Loader
    ↓
Document Chunking
    ↓
Gemini Embeddings
    ↓
FAISS Vector Database
    ↓
       ┌──────────────┐
Query →│ FAISS Search │
       └──────────────┘
              +
       ┌──────────────┐
       │ BM25 Search  │
       └──────────────┘
              ↓
       Hybrid Retrieval
              ↓
          Reranking
              ↓
      Relevant Documents
              ↓
     Conversation Memory
              ↓
        Gemini LLM
              ↓
       Grounded Answer
              ↓
        Streamlit UI

---

4. Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Application development |
| LangChain | RAG components |
| Google Gemini | Embeddings and LLM |
| FAISS | Vector database |
| BM25 | Keyword retrieval |
| Sentence Transformers | Cross-encoder reranking |
| Streamlit | User interface |
| PyPDF | PDF processing |
| python-docx | DOCX processing |

---

5. Project Structure

```text
Project AI/
│
├── data/
│   ├── documents/
│   └── embeddings/
│       └── faiss_gemini/
│
├── src/
│   ├── ingestion/
│   │   ├── Document_loader.py
│   │   └── Chunker.py
│   │
│   ├── embeddings/
│   │   └── embedding_model.py
│   │
│   ├── retrieval/
│   │   ├── BM25_retriever.py
│   │   ├── faiss_retriever.py
│   │   └── hybrid_retriever.py
│   │
│   ├── reranking/
│   │   └── reranker.py
│   │
│   ├── memory/
│   │   └── conversation_memory.py
│   │
│   ├── generation/
│   │   └── llm_generator.py
│   │
│   └── pipeline/
│       └── rag_pipeline.py
│
├── app.py
├── config.yaml
├── requirements.txt
├── .env
└── README.md

---

6. Document Ingestion

The system supports multiple document formats:

PDF
DOCX
TXT

Documents are loaded using format-specific loaders and converted into LangChain Document objects.

PDF text is normalized to handle whitespace introduced during PDF extraction.

---

7. Chunking

Large documents are divided into smaller chunks using a recursive text splitter.

Chunking improves retrieval because the system can retrieve only the relevant portions of a document instead of processing the complete document.

---

8. Embeddings and Vector Search

Google Gemini embedding models convert document chunks into numerical vectors.

These vectors are stored in FAISS.

During a query, the user's question is converted into an embedding and compared with stored document vectors to retrieve semantically relevant information.

---

9. Hybrid Retrieval

The system combines two retrieval techniques:

FAISS

Used for semantic similarity search.

BM25

Used for keyword-based retrieval.

Combining both methods improves retrieval because semantic search understands meaning while BM25 performs well for exact terms and keywords.

---

10. Reranking

The retrieved documents are passed through a Cross-Encoder reranker.

The reranker evaluates the relevance between:

User query
Retrieved document

The most relevant documents are then selected as context for the LLM.

---

11. Conversational Memory

The system maintains recent conversation history for each session.

This allows the assistant to understand follow-up questions.

Example:

User:
"What is the probation period?"

Assistant:
"The normal probation period is six months."

User:
"Is it the same for everyone?"

The system uses the previous question together with the new question during retrieval.

---

12. Hallucination Handling

The LLM is instructed to answer only using the retrieved document context.

If sufficient information is not available, the system responds:

"I don't have enough information in the provided documents."

Example:

Question:
"What is the company's maternity leave policy?"

Response:
"I don't have enough information in the provided documents."

This prevents the system from inventing unsupported information.

---

13. Example Test Results
Q
How many annual leave days do full-time employees get?	- 18 days - PASS
How many sick leave days do employees get? - 10 days - PASS
What is the probation period? - 6 months - PASS
What is the maternity leave policy? - Information unavailable - PASS
Is the probation period the same for everyone? - Insufficient information - PASS
What is my name? - Uses conversation memory - PASS

---

14. Running the Application

Activate the virtual environment:

source myenv/bin/activate

Set the Gemini API key in .env:

GEMINI_API_KEY=your_api_key

Run the application:

streamlit run app.py

The application opens in the browser.

---

15. Conclusion

The TechNova Knowledge Assistant demonstrates an end-to-end Advanced RAG architecture combining semantic retrieval, keyword retrieval, reranking, conversational memory and grounded generation.

The system provides accurate answers from enterprise documents while reducing hallucinations when information is unavailable.


### 2. Save it

Your final project is now in a good submission state.

**Next, I recommend we prepare your 5–7 minute viva explanation and likely questions/answers.**