from crewai import Agent, Task, Crew
from crewai.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM - Using ChatOpenAI instead of OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-4" for better results
    temperature=0, 
    api_key=os.getenv("OPENAI_API_KEY")
)

# Ingest PDFs if not already indexed
if not os.path.exists("law_index"):
    loader = PyPDFDirectoryLoader("lawdata")  # <- Load all PDFs from the 'lawdata' directory
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("law_index")
else:
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.load_local("law_index", embeddings, allow_dangerous_deserialization=True)

# Define a retriever tool
@tool
def retrieve_info(query: str) -> str:
    """Search legal documents for relevant information."""
    docs = db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

# Define agents
retriever = Agent(
    role="Legal Information Retriever",
    goal="Find relevant information from legal documents based on user queries",
    backstory="You are an expert legal researcher who excels at finding precise and relevant legal information from documents. You have years of experience in legal research and case law analysis.",
    tools=[retrieve_info],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

summarizer = Agent(
    role="Legal Information Summarizer", 
    goal="Condense complex legal information into clear, accurate summaries",
    backstory="You are a legal communication specialist who excels at distilling complex legal and technical content into clear, understandable summaries while maintaining legal accuracy.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

answerer = Agent(
    role="Legal Answer Generator",
    goal="Generate comprehensive, user-friendly legal answers based on retrieved information",
    backstory="You are a legal educator who specializes in explaining legal concepts clearly to clients and the general public, always emphasizing the importance of consulting qualified attorneys for legal advice.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

def process_legal_query(query: str):
    """Process a legal query through the crew workflow"""
    
    # Create tasks with better descriptions and expected outputs
    task1 = Task(
        description=f"Search the legal documents for information related to: '{query}'. Focus on finding the most relevant and accurate legal information, statutes, cases, or regulations.",
        agent=retriever,
        expected_output="Relevant legal information from the documents"
    )
    
    task2 = Task(
        description=f"Summarize the retrieved legal information about '{query}' into key points. Focus on accuracy and clarity while maintaining legal precision.",
        agent=summarizer,
        expected_output="A clear summary of the legal information",
        context=[task1]  # This task depends on task1
    )
    
    task3 = Task(
        description=f"Generate a comprehensive, user-friendly answer to the legal question: '{query}'. Include appropriate disclaimers about consulting qualified attorneys.",
        agent=answerer,
        expected_output="A complete, user-friendly legal answer with appropriate disclaimers",
        context=[task1, task2]  # This task depends on both previous tasks
    )
    
    # Create and run the crew
    crew = Crew(
        agents=[retriever, summarizer, answerer],
        tasks=[task1, task2, task3], 
        verbose=True,
        process="sequential"  # Explicitly set the process type
    )
    
    return crew.kickoff()

# Main execution
if __name__ == "__main__":
    print("⚖️ Legal Q&A System - Ask a legal question (type 'exit' to quit):\n")
    
    while True:
        query = input("📚 You: ")
        if query.strip().lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye! Remember to always consult with qualified attorneys for legal advice.")
            break
        
        if not query.strip():
            print("Please enter a valid question.\n")
            continue
            
        try:
            print("\n🔍 Processing your legal query...\n")
            result = process_legal_query(query)
            print(f"\n✅ Final Answer:\n{result}\n")
            print("-" * 50 + "\n")
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            print("Please try again with a different question.\n")