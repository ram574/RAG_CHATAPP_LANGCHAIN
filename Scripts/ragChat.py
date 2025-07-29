import os
import sys
import logging
from typing import Literal
from typing_extensions import Annotated, List, TypedDict

from PyPDF2 import PdfMerger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langgraph.graph import START, StateGraph

from dotenv import load_dotenv
load_dotenv("../.env", override=True)

# --- Configuration ---
PDF_FOLDER = "/home/tulasiram/ubuntu_server/RAG_ChatApp_LangChain/Input_Data/health-plan"
OUTPUT_PDF_PATH = "/home/tulasiram/ubuntu_server/RAG_ChatApp_LangChain/Output_Data/merged_output.pdf"
VECTOR_INDEX_PATH = "/home/tulasiram/ubuntu_server/RAG_ChatApp_LangChain/vector_index"

AZURE_EMBEDDING_CONFIG = {
    "azure_endpoint": "https://25996-openai.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15",
    "api_key": "3b578c1bce87445fad51ca5b03212249",
    "api_version": "2023-05-15",
    "deployment": "text-embedding-ada-002",
    "model": "text-embedding-ada-002",
}

AZURE_CHAT_CONFIG = {
    "azure_endpoint": "https://25996-gpt-4o-mini.openai.azure.com/",
    "api_key": "770082685af343f59fd16883fa006c48",
    "deployment_name": "gpt-4o-mini",
    "api_version": "2025-01-01-preview",
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 1000,
}

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Preparation Functions ---
def merge_pdfs(pdf_folder: str, output_pdf_path: str):
    """Merge all PDFs in the folder into a single PDF."""
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        logger.error("No PDF files found in the input folder.")
        sys.exit(1)
    merger = PdfMerger()
    for pdf in pdf_files:
        merger.append(pdf)
    merger.write(output_pdf_path)
    merger.close()
    logger.info(f"Merged {len(pdf_files)} PDFs into {output_pdf_path}")

def load_and_split_pdf(pdf_path: str, chunk_size=850, chunk_overlap=100):
    """Load PDF and split into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages from merged file.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(documents)
    logger.info(f"Split the merged document into {len(all_splits)} sub-documents.")
    return all_splits

def add_section_metadata(all_splits: List[Document]):
    """Add section metadata to document chunks."""
    total_documents = len(all_splits)
    third = total_documents // 3
    for i, document in enumerate(all_splits):
        if i < third:
            document.metadata["section"] = "beginning"
        elif i < 2 * third:
            document.metadata["section"] = "middle"
        else:
            document.metadata["section"] = "end"
    logger.info("Added section metadata to document chunks.")

# --- RAG Pipeline Classes ---
class Search(TypedDict):
    """Search query."""
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

# --- RAG Pipeline Functions ---
def analyze_query(state: State, llm):
    """Analyze the question and produce a structured search query."""
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

def retrieve(state: State, vectorstore):
    """Retrieve relevant documents based on the query and section."""
    query = state["query"]
    retrieved_docs = vectorstore.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}

def generate(state: State, prompt, llm):
    """Generate an answer using the retrieved context."""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# --- Main Execution ---
def main():
    # Step 1: Merge PDFs
    merge_pdfs(PDF_FOLDER, OUTPUT_PDF_PATH)

    # Step 2: Load and split PDF
    all_splits = load_and_split_pdf(OUTPUT_PDF_PATH)
    add_section_metadata(all_splits)

    # Step 3: Embedding and vector store
    embedding_model = AzureOpenAIEmbeddings(**AZURE_EMBEDDING_CONFIG)
    vectorstore = InMemoryVectorStore(embedding_model)
    _ = vectorstore.add_documents(all_splits)

    # Step 4: LLM and prompt setup
    llm = AzureChatOpenAI(**AZURE_CHAT_CONFIG)
    prompt = hub.pull("rlm/rag-prompt")
    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    # Step 5: Build graph
    def analyze_query_node(state: State):
        return analyze_query(state, llm)

    def retrieve_node(state: State):
        return retrieve(state, vectorstore)

    def generate_node(state: State):
        return generate(state, custom_rag_prompt, llm)

    graph_builder = StateGraph(State).add_sequence([analyze_query_node, retrieve_node, generate_node])
    graph_builder.add_edge(START, "analyze_query_node")
    graph = graph_builder.compile()

    # Step 6: User interaction
    question = input("Enter your question: ")
    for step in graph.stream(
        {"question": question},
        stream_mode="updates",
        ):
            print(f"{step}\n\n----------------\n")

if __name__ == "__main__":
    main()