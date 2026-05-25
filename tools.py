import json
from difflib import SequenceMatcher
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
from lib.documents import Document
from lib.tooling import Tool
from lib.memory import LongTermMemory, MemoryFragment

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

rag_llm = LLM(model="gpt-4o-mini", temperature=0.3)
vector_store_manager = VectorStoreManager(os.getenv("OPENAI_API_KEY"))
games_data_vector_store = vector_store_manager.create_store(store_name="games_data", force=True)
long_term_memory = LongTermMemory(vector_store_manager)


@tool
def get_games(num_games:int=1, top:bool=True) -> str:
    """
    Returns the top or bottom N games with highest or lowest scores.    
    args:
        num_games (int): Number of games to return (default is 1)
        top (bool): If True, return top games, otherwise return bottom (default is True)
    """
    with open('games.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Sort the games list by Score
        # If top is True, descending order
        sorted_games = sorted(data, key=lambda x: x['Score'], reverse=top)

        # Return the N games
        return str(sorted_games[:num_games])
    
games_data_vector_store.add(Document(content=get_games(num_games=10, top=True)))
games_rag = RAG(llm=rag_llm, vector_store=games_data_vector_store)

@tool
def retrieve_game(game_name:str) -> QueryResult:
    """
    Retrieves game information from the vector store based on the game name.
    args:
        game_name (str): The name of the game to retrieve
    returns:
        QueryResult: The result of the query
    """
    return games_rag.invoke(game_name).get_final_state()["answer"]

@tool
def evaluate_retrieval(found_game_name:str, expected_game_name:str) -> float:
    """
    Evaluates the retrieval result by comparing the retrieved game name with the expected game name.
    args:
        found_game_name (str): The result of the retrieval query
        expected_game_name (str): The expected game name to compare against
    returns:
        float: how similar the retrieved game name is to the expected game name in percentage
    """
    if not found_game_name:
        return 0.0  # No documents retrieved, so similarity is 0%
    

    similarity = SequenceMatcher(
        None,
        found_game_name,
        expected_game_name
    ).ratio()
    return round(similarity * 100, 2)

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
        include_images=False
    )
    
    # Format the results
    formatted_results = {
        "answer": search_result.get("answer", ""),
        "results": search_result.get("results", []),
        "search_metadata": {
            "timestamp": datetime.now().isoformat(),
            "query": query
        }
    }
    
    return formatted_results

def build_memory_registration_tool(ltm:LongTermMemory, owner:str, namespace:str):
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
    def _register(content:str):
        ltm.register(
            MemoryFragment(
                content=content, 
                owner=owner,
                namespace=namespace
            )
        )
        return "Saved new memory"

    return Tool(
        func=_register, 
        name="register_memory", 
        description=(
            "Register a new memory about the information of the game for future reference. "
            "Args:\n"
            "    content: The information to save"
        )
    )

def build_memory_search_tool(ltm:LongTermMemory, owner:str, namespace:str):
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
    def _search(content:str):
        result = ltm.search(
            query_text=content,
            owner=owner,
            namespace=namespace,
            limit=3,
        )
        return str(tuple(zip(result.fragments, result.metadata['distances'])))

    return Tool(
        func=_search, 
        name="search_memory", 
        description=(
            "Search for a stored memory or preference about the game, " 
            "so it's useful as a context.\n"
            "Args:\n"
            "    content: The information to look for"
        )
    )
