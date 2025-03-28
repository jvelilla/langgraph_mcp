#MCP server

from mcp.server.fastmcp import FastMCP
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from pathlib import Path

# Define common path to the repo locally
PATH = "C:/home/Learn/rag_dev/mcp/"

# Create an MCP server
mcp = FastMCP("LangGraph-Docs-MCP-Server")

# Add a tool to query the LangGraph documentation
@mcp.tool()
def langgraph_query_tool(query: str):
    """
    Query the LangGraph documentation using a retriever.
    
    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    retriever = SKLearnVectorStore(
        embedding=OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434"), 
        persist_path=PATH + "sklearn_vectorstore.parquet", 
        serializer="parquet").as_retriever(search_kwargs={"k": 3}
        )

    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")
    formatted_context = "\n\n".join([f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    return formatted_context

# The @mcp.resource() decorator is meant to map a URI pattern to a function that provides the resource content
@mcp.resource("docs://langgraph/full")
def get_all_langgraph_docs() -> str:
    """
    Get all the LangGraph documentation. Returns the contents of the file llms_full.txt,
    which contains a curated set of LangGraph documentation (~300k tokens). This is useful
    for a comprehensive response to questions about LangGraph.

    Args: None

    Returns:
        str: The contents of the LangGraph documentation
    """

    # Local path to the LangGraph documentation
    doc_path = PATH + "llms_full.txt"
    try:
        with open(doc_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading log file: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')