# 🤖 TechNova Knowledge Assistant – Enterprise Advanced RAG System

## 1. Project Overview

**TechNova Knowledge Assistant** is an Enterprise Knowledge Assistant built using an advanced **Retrieval-Augmented Generation (RAG)** architecture.

The system is designed to retrieve accurate information from internal company documents such as employee handbooks, leave policies, and employee benefits documents.

It supports **PDF, DOCX, and TXT** documents and combines:

* Document ingestion
* Text chunking
* Gemini embeddings
* FAISS semantic retrieval
* BM25 keyword retrieval
* Hybrid retrieval
* Cross-Encoder reranking
* Conversational memory
* Multi-chat sessions
* Grounded Gemini LLM generation
* Hallucination handling
* Streamlit-based user interface

The system is designed to reduce hallucinations by answering questions only from the retrieved document context. When the required information is unavailable, the system informs the user instead of generating unsupported information.

---

## 2. Live Demo

**Streamlit Application:**

https://technova-knowledge-assistant-xlcpzgamd3cj3lvctuwgke.streamlit.app/

**GitHub Repository:**

https://github.com/Bavana2311/technova-knowledge-assistant

---

## 3. Key Features

* Multi-format document ingestion
* PDF, DOCX, and TXT support
* Document chunking
* Gemini embeddings
* FAISS vector database
* BM25 keyword-based retrieval
* Hybrid retrieval
* Cross-Encoder reranking
* Conversational memory
* Context-aware follow-up questions
* Multiple chat sessions
* Session-specific conversation memory
* Hallucination handling
* Grounded Gemini LLM answer generation
* Streamlit user interface

---

## 4. System Architecture


                    Enterprise Documents
                            │
                            ▼
                    Document Loader
                            │
                            ▼
                     Text Chunking
                            │
                            ▼
                    Gemini Embeddings
                            │
                            ▼
                    FAISS Vector Store
                            │
                            │
                    ┌───────┴────────┐
                    │                │
                    ▼                ▼
              FAISS Search      BM25 Search
             Semantic Search   Keyword Search
                    │                │
                    └───────┬────────┘
                            ▼
                    Hybrid Retrieval
                            │
                            ▼
                       Reranking
                            │
                            ▼
                  Relevant Documents
                            │
                            ▼
                 Conversation Memory
                            │
                            ▼
                       Gemini LLM
                            │
                            ▼
                   Grounded Answer
                            │
                            ▼
                      Streamlit UI


---

## 5. Technologies Used

| Technology            | Purpose                          |
| --------------------- | -------------------------------- |
| Python                | Application development          |
| LangChain             | RAG pipeline components          |
| Google Gemini         | LLM and embeddings               |
| Gemini Embeddings     | Document and query vectorization |
| FAISS                 | Semantic vector search           |
| BM25                  | Keyword-based retrieval          |
| Sentence Transformers | Cross-Encoder reranking          |
| Streamlit             | Web-based user interface         |
| PyPDF                 | PDF document processing          |
| python-docx           | DOCX document processing         |
| YAML                  | Application configuration        |
| python-dotenv         | Environment variable management  |

---

## 6. Project Structure


Advanced RAG Pipeline/
│
├── data/
│   └── documents/
│       ├── Employee_benefits.txt
│       ├── Employee_Leave_Policy.pdf
│       └── Employee_Handbook.docx
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
├── runtime.txt
├── template.py
├── test.ipynb
├── .gitignore
└── README.md


> **Note:** The `.env` file contains the Gemini API key and should remain private. It should not be committed to GitHub.

---

## 7. Document Ingestion

The system supports multiple document formats:

* PDF
* DOCX
* TXT

Documents are loaded using format-specific document loaders and converted into LangChain `Document` objects.

The current knowledge base contains:

* Employee Handbook
* Employee Leave Policy
* Employee Benefits

The document ingestion layer provides the source content used by the retrieval pipeline.

---

## 8. Document Chunking

Large documents are divided into smaller chunks using a recursive text-splitting approach.

Chunking improves retrieval because the system can retrieve relevant portions of documents instead of processing the complete document for every query.

This allows the retrieval system to focus on smaller and more relevant pieces of information.

---

## 9. Embeddings and Vector Search

Google Gemini embedding models convert document chunks into numerical vector representations.

These embeddings are stored in **FAISS**, which performs semantic similarity search.

During a user query:

```text
User Question
      ↓
Gemini Query Embedding
      ↓
FAISS Similarity Search
      ↓
Semantically Relevant Chunks
```

This enables the system to retrieve information based on meaning rather than relying only on exact keyword matches.

---

## 10. Hybrid Retrieval

The system combines two retrieval techniques:

### FAISS

FAISS performs semantic similarity search using Gemini embeddings.

It is useful when the query and document use different wording but have the same meaning.

### BM25

BM25 performs keyword-based retrieval.

It is useful for exact terms, names, policy terminology, and other keyword-heavy queries.

### Hybrid Retrieval

The two retrieval methods are combined to improve the overall retrieval process.


                 User Query
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     FAISS Search           BM25 Search
     Semantic Search        Keyword Search
          │                     │
          └──────────┬──────────┘
                     ▼
              Hybrid Retrieval


---

## 11. Cross-Encoder Reranking

The retrieved documents are passed through a **Cross-Encoder reranker**.

The reranker evaluates the relevance between:

```text
User Query
     +
Retrieved Document
```

The most relevant documents are then selected as context for the Gemini LLM.

This provides an additional relevance-filtering stage after the initial retrieval process.

---

## 12. Conversational Memory

The system maintains conversation history for each chat session.

This allows the assistant to understand follow-up questions using previous conversation context.

### Example

**User:**

> What is the probation period?

**Assistant:**

> The normal probation period is six months.

**User:**

> Is it the same for everyone?

The system can use the previous conversation together with the new question to interpret the follow-up.

---

## 13. Multiple Chat Sessions

The Streamlit interface supports multiple independent chat sessions.

Each chat session has:

* A unique session ID
* Its own conversation history
* Its own memory state

Example:

```text
Chat 1
 ├── User: My name is Bavana
 ├── Assistant: Nice to meet you
 └── User: What is my name?
     └── Assistant: Your name is Bavana

Chat 2
 └── New conversation
```

Starting a new chat creates a separate session instead of mixing the conversation history with previous chats.

---

## 14. Hallucination Handling

The LLM is instructed to answer using the retrieved document context.

If sufficient information is not available in the provided documents, the system responds:

> "I don't have enough information in the provided documents."

### Example

**Question:**

> What is the company's maternity leave policy?

**Response:**

> I don't have enough information in the provided documents.

This approach helps prevent the system from generating unsupported information.

---

## 15. Example Test Results

| Question                                               | Expected Result                  | Status |
| ------------------------------------------------------ | -------------------------------- | ------ |
| How many annual leave days do full-time employees get? | 18 days                          | PASS   |
| How many sick leave days do employees get?             | 10 days                          | PASS   |
| What is the probation period?                          | 6 months                         | PASS   |
| What is the maternity leave policy?                    | Information unavailable          | PASS   |
| Is the probation period the same for everyone?         | Insufficient information         | PASS   |
| What is my name?                                       | Uses conversation memory         | PASS   |
| Summarize our conversation so far.                     | Summarizes previous conversation | PASS   |
| New Chat                                               | Creates independent session      | PASS   |

---

## 16. Running the Application

### Step 1: Clone the repository

```bash
git clone https://github.com/Bavana2311/technova-knowledge-assistant.git
cd technova-knowledge-assistant
```

### Step 2: Create and activate the virtual environment

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure the Gemini API key

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key
```

Do not commit the `.env` file to GitHub.

### Step 5: Run the application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 17. Configuration

The main configuration is maintained in `config.yaml`.

It contains settings for:

* Gemini LLM
* Gemini embeddings
* FAISS
* Hybrid retrieval
* BM25
* Cross-Encoder reranking

This separates model and retrieval configuration from the application code.

---

## 18. RAG Pipeline Flow

The complete question-answering process is:

```text
User Question
      ↓
Query Processing
      ↓
FAISS Semantic Retrieval
      +
BM25 Keyword Retrieval
      ↓
Hybrid Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Relevant Context
      ↓
Conversation Memory
      ↓
Gemini LLM
      ↓
Grounded Response
      ↓
Streamlit Chat Interface
```

---

## 19. Advantages of the Approach

The architecture combines multiple techniques instead of relying on a single retrieval method.

### Semantic Retrieval

Helps identify documents based on meaning.

### Keyword Retrieval

Helps identify exact terms and keywords.

### Reranking

Provides an additional relevance evaluation stage.

### Conversational Memory

Allows the assistant to understand context across questions.

### Grounded Generation

Restricts answers to retrieved document information and provides a fallback when information is unavailable.

### Multi-Chat Support

Allows users to maintain separate conversations without mixing session histories.

---

## 20. Conclusion

The **TechNova Knowledge Assistant** demonstrates an end-to-end Enterprise Advanced RAG architecture combining:

* Multi-format document ingestion
* Document chunking
* Gemini embeddings
* FAISS semantic retrieval
* BM25 keyword retrieval
* Hybrid retrieval
* Cross-Encoder reranking
* Conversational memory
* Multiple chat sessions
* Grounded Gemini generation
* Hallucination handling
* Streamlit deployment

The system provides a practical enterprise knowledge-assistant workflow for retrieving information from internal documents while reducing unsupported or hallucinated responses.

This project demonstrates the implementation of an advanced RAG pipeline from document ingestion through retrieval, reranking, memory, and grounded answer generation.
