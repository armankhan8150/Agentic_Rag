# Agentic RAG Legal Q&A

**A “crew” of AI agents** for retrieval-augmented legal question-answering, built on CrewAI + LangChain + FAISS.

## 🚀 Features
- **Automatic Ingestion** of all PDFs in `data_ingest` (your “lawdata” folder)
- **Chunking & Embedding** via LangChain’s `CharacterTextSplitter` and OpenAIEmbeddings
- **Vector Store** with FAISS (saved to `law_index/`)
- **Three AI Agents**  
  1. **Retriever** – fetches the top‐k relevant chunks  
  2. **Summarizer** – distills them into key points  
  3. **Answerer** – crafts a user‐friendly, legally precise reply (with disclaimers)

## 📦 Prerequisites
- Python 3.10+  
- An OpenAI API key with permission for embeddings & Chat (v4-compatible models)  
- `pip install -r requirements.txt`

## 🔧 Setup

1. **Clone and install**
 git clone <your-repo-url>
 cd <repo-folder>
 pip install -r requirements.txt

2. **Configure your environment**
Create a .env in the root:
  OPENAI_API_KEY=sk-… 

3. **Ingest your PDFs**
Put all your .pdf files into data_ingest
On first run, agentic_rag.py will build law_index/.

**Usage**
python agentic_rag.py
You'll see: 
⚖️ Legal Q&A System – Ask a legal question (type 'exit' to quit):
📚 You:  What is the statute of limitations for contract disputes?

**The script will:**
Build/search FAISS index
Retrieve, summarize, answer
Print the final response
