import json
from lib.tooling import tool
from lib.vector_db import VectorStore
from tavily import TavilyClient
from chromadb.api.types import QueryResult
from typing import Dict
from datetime import datetime
from lib.llm import LLM
from lib.rag import RAG
import os
from lib.vector_db import VectorStoreManager
from lib.documents import Document, Corpus
from lib.tooling import Tool
from lib.memory import LongTermMemory, MemoryFragment
from lib.evaluation import EvaluationReport
from lib.parsers import PydanticOutputParser

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

rag_llm = LLM(model="gpt-4o-mini", temperature=0.3)
CHROMA_PERSIST_DIR = os.path.join(os.path.abspath("."), "chroma_db")
vector_store_manager = VectorStoreManager(
    os.getenv("OPENAI_API_KEY"), persist_directory=CHROMA_PERSIST_DIR
)
long_term_memory = LongTermMemory(vector_store_manager)

def _load_game_documents():
    games_dir = os.path.join(os.path.abspath("."), "games")
    documents = []
    for filename in sorted(os.listdir(games_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(games_dir, filename), "r", encoding="utf-8") as f:
                g = json.load(f)
            documents.append(
                Document(
                    content=(
                        f"{g['Name']} is available on {g['Platform']}. "
                        f"Genre: {g.get('Genre', 'Unknown')}. "
                        f"Publisher: {g.get('Publisher', 'Unknown')}. "
                        f"Year of release: {g.get('YearOfRelease', 'Unknown')}. "
                        f"{g.get('Description', '')}"
                    ),
                    metadata={
                        "game": g["Name"],
                        "platform": g["Platform"],
                        "genre": g.get("Genre", ""),
                        "publisher": g.get("Publisher", ""),
                        "year": g.get("YearOfRelease", ""),
                    },
                )
            )
    return Corpus(documents)


long_term_memory.vector_store.add(_load_game_documents())
games_rag = RAG(llm=rag_llm, vector_store=long_term_memory.vector_store)


@tool
def retrieve_game(game_name: str) -> QueryResult:
    """
    Retrieves game information from the vector store based on the game name.
    args:
        game_name (str): The name of the game to retrieve
    returns:
        QueryResult: The result of the query
    """
    return games_rag.invoke(game_name).get_final_state()["answer"]


@tool
def evaluate_retrieval(question: str, retrieved_docs: str) -> dict:
    """
    Based on the user's question and on the list of retrieved documents,
    it will analyze the usability of the documents to respond to that question.
    args:
        question (str): original question from user
        retrieved_docs (str): retrieved documents most similar to the user query in the Vector Database
    returns:
        dict with:
        - useful: whether the documents are useful to answer the question
        - description: description about the evaluation result
    """
    judge_llm = LLM(model="gpt-4o-mini", temperature=0.0)
    prompt = (
        "Your task is to evaluate if the documents are enough to respond the query. "
        "Give a detailed explanation, so it's possible to take an action to accept it or not.\n\n"
        f"Query: {question}\n\n"
        f"Retrieved Documents:\n{retrieved_docs}"
    )
    response = judge_llm.invoke(input=prompt, response_format=EvaluationReport)
    parser = PydanticOutputParser(model_class=EvaluationReport)
    result = parser.parse(response)
    return {"useful": result.useful, "description": result.description}


@tool
def game_web_search(query: str, search_depth: str = "advanced") -> Dict:
    """
    Search the web using Tavily API
    args:
        query (str): Search query
        search_depth (str): Type of search - 'basic' or 'advanced' (default: advanced)
    """

    # Perform the search
    search_result = tavily_client.search(
        query=query,
        search_depth=search_depth,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )

    # Format the results
    formatted_results = {
        "answer": search_result.get("answer", ""),
        "results": search_result.get("results", []),
        "citations": [
            {"title": r.get("title", ""), "url": r.get("url", "")}
            for r in search_result.get("results", [])
        ],
        "search_metadata": {"timestamp": datetime.now().isoformat(), "query": query},
    }

    return formatted_results


def build_memory_registration_tool(ltm: LongTermMemory, owner: str, namespace: str):
    """
    Create a tool for agents to register new memories.

    This factory function creates a tool that allows AI agents to store new
    information about users in the long-term memory system. The tool is
    pre-configured with specific owner and namespace parameters.

    Args:
        ltm (LongTermMemory): The memory system instance to use
        owner (str): User identifier for memory ownership
        namespace (str): Namespace for organizing memories

    Returns:
        Tool: A configured tool for memory registration
    """

    def _register(content: str):
        ltm.register(MemoryFragment(content=content, owner=owner, namespace=namespace))
        return "Saved new memory"

    return Tool(
        func=_register,
        name="register_memory",
        description=(
            "Register a new memory about the information of the game for future reference. "
            "Args:\n"
            "    content: The information to save"
        ),
    )


def build_memory_search_tool(ltm: LongTermMemory, owner: str, namespace: str):
    """
    Create a tool for agents to search existing memories.

    This factory function creates a tool that allows AI agents to retrieve
    relevant memories from the long-term memory system based on semantic
    similarity to a search query.

    Args:
        ltm (LongTermMemory): The memory system instance to use
        owner (str): User identifier for memory ownership
        namespace (str): Namespace to search within

    Returns:
        Tool: A configured tool for memory search
    """

    def _search(content: str):
        result = ltm.search(
            query_text=content,
            owner=owner,
            namespace=namespace,
            limit=3,
        )
        return str(tuple(zip(result.fragments, result.metadata["distances"])))

    return Tool(
        func=_search,
        name="search_memory",
        description=(
            "Search for a stored memory or preference about the game, "
            "so it's useful as a context.\n"
            "Args:\n"
            "    content: The information to look for"
        ),
    )
