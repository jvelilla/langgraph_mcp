# LangGraph Documentation MCP Server

This project implements a Mod Control Protocol (MCP) server that provides access to LangGraph documentation through a vector store-based retrieval system. The implementation is based on the [MCP From Scratch tutorial](https://mirror-feeling-d80.notion.site/MCP-From-Scratch-1b9808527b178040b5baf83a991ed3b2?pvs=4) and has been updated to use Ollama for embeddings.

## Features

- Vector store-based document retrieval using SKLearnVectorStore
- Ollama embeddings for document vectorization
- MCP server implementation with FastMCP
- Document loading and processing from LangGraph documentation
- Support for both tool-based queries and full documentation access

## Prerequisites

- Python 3.12+
- Ollama installed and running locally (default port: 11434)
- Required Python packages 
    - langchain_community langchain-anthropic langchain_ollama scikit-learn bs4 pandas pyarrow matplotlib lxml langgraph tiktoken "mcp[cli]

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install langchain_community langchain-anthropic langchain_ollama scikit-learn bs4 pandas pyarrow matplotlib lxml langgraph tiktoken "mcp[cli]"
```

## Project Structure

- `langgraph_mcp.py`: Main MCP server implementation
- `build-tool.ipynb`: Jupyter notebook for building and testing the vector store
- `llms_full.txt`: Generated documentation file
- `sklearn_vectorstore.parquet`: Vector store file

## Usage

1. First, build the vector store using the Jupyter notebook:
```bash
jupyter notebook build-tool.ipynb
```

2. Run the MCP server:
```bash
python langgraph_mcp.py
```

## Available Tools

### langgraph_query_tool
A tool that queries the LangGraph documentation using semantic search:
```python
@mcp.tool()
def langgraph_query_tool(query: str):
    """
    Query the LangGraph documentation using a retriever.
    
    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
```

### Full Documentation Access
Access the complete LangGraph documentation through the resource endpoint:
```python
@mcp.resource("docs://langgraph/full")
def get_all_langgraph_docs() -> str:
    """
    Get all the LangGraph documentation.
    """
```

## Implementation Details

The project uses:
- Ollama embeddings with the `nomic-embed-text` model
- SKLearnVectorStore for document storage and retrieval
- BeautifulSoup for HTML parsing
- RecursiveUrlLoader for documentation scraping
- RecursiveCharacterTextSplitter for document chunking

## Credits

This implementation is based on:
- [MCP From Scratch Tutorial](https://mirror-feeling-d80.notion.site/MCP-From-Scratch-1b9808527b178040b5baf83a991ed3b2?pvs=4)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama](https://ollama.ai/)

## License

This project is open source and available under the MIT License. 