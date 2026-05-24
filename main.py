from dotenv import load_dotenv
import os

from tools import retrieve_game,evaluate_retrieval,game_web_search

from lib.agents import Agent
from lib.messages import BaseMessage
from typing import List


def _print_messages(messages: List[BaseMessage]):
    for m in messages:
        print(f" -> (role = {m.role}, content = {m.content}, tool_calls = {getattr(m, 'tool_calls', None)})")

def setup_env():
    load_dotenv("config.env")
    assert os.getenv("OPENAI_API_KEY") is not None
    assert os.getenv("TAVILY_API_KEY") is not None


if __name__ == "__main__":
    setup_env()

    # db = VectorStoreManager(os.getenv("OPENAI_API_KEY"))

    # rag_llm = LLM(model="gpt-4o-mini", temperature=0.3)

    # vector_store_manager = VectorStoreManager(os.getenv("OPENAI_API_KEY"))
    # games_data_vector_store = vector_store_manager.create_store(store_name="games_data", force=True)
    # games_data_vector_store.add(Document(content=get_games(num_games=10, top=True)))

    # games_rag = RAG(llm=rag_llm, vector_store=games_data_vector_store)
    
    # run=games_rag.invoke("What are the top 3 games?")
    # print(run.get_final_state()["answer"])

    my_agent=Agent(model_name="gpt-4o-mini", tools=[retrieve_game,evaluate_retrieval,game_web_search],temperature=0.3,instructions="You are a helpful assistant for game recommendations and information retrieval." \
    " Use the tools at your disposal to answer user queries about games. Don't use your training data. Always try to use the tools when relevant information can be retrieved or actions can be taken. If you use a tool, make sure to incorporate the results into your final response to the user."
    " When you look up a game and find a result, evaluate how well it matches the user's query if the similarity is above 75% fall back to web search tool to find more information about the game and provide a comprehensive answer to the user.")

    run1 = my_agent.invoke(
    query="give me score and platform info about Super Smash Bros", 
    )

    print("\nMessages from run 1:")
    messages = run1.get_final_state()["messages"]
    _print_messages(messages)