from dotenv import load_dotenv
import os

from tools import (
    retrieve_game,
    evaluate_retrieval,
    game_web_search,
    long_term_memory,
    build_memory_registration_tool,
    build_memory_search_tool,
)

from lib.agents import Agent
from lib.messages import BaseMessage
from typing import List


def _print_messages(messages: List[BaseMessage]):
    for m in messages:
        print(
            f" -> (role = {m.role}, content = {m.content}, tool_calls = {getattr(m, 'tool_calls', None)})"
        )


def setup_env():
    load_dotenv("config.env")
    assert os.getenv("OPENAI_API_KEY") is not None
    assert os.getenv("TAVILY_API_KEY") is not None


if __name__ == "__main__":
    setup_env()

    my_agent = Agent(
        model_name="gpt-4o-mini",
        tools=[
            retrieve_game,
            evaluate_retrieval,
            build_memory_search_tool(long_term_memory, "omer_oksuzler", "games"),
            game_web_search,
            build_memory_registration_tool(long_term_memory, "omer_oksuzler", "games"),
        ],
        temperature=0.3,
        instructions=(
            "You are a helpful assistant for game recommendations and information retrieval. "
            "When answering a query about a game, you MUST follow this exact tool-use order:\n"
            "1. Call retrieve_game first to search the local game database.\n"
            "1.1 if a game is found, call evaluate_retrieval to check if the retrieved game information is relevant and sufficient to answer the user's query. If evaluate_retrieval returns useful you can persist the memory and finish\n"
            "2. Only if retrieve_game returns no result or insufficient information, call the long-term memory search tool to check previously stored knowledge.\n"
            "3. Only if the long-term memory search also returns no useful result, call game_web_search to find the information online.\n"
            "4. After using game_web_search, you MUST always call the long-term memory registration tool to persist the retrieved information for future use. And Add citations to every final answer that uses web search, not just to the raw tool output.\n"
            "Never skip steps or jump ahead. Always incorporate all retrieved information into your final response. Be concise and informative."
        ),
    )

    run1 = my_agent.invoke(
        query="When Pokémon Gold and Silver was released?", session_id="session_1"
    )

    run2 = my_agent.invoke(
        query="Which one was the first 3D platformer Mario game?", session_id="session_2"
    )

    run3 = my_agent.invoke(
        query="Was Mortal Kombat X realeased for Playstation 5?", session_id="session_2"
    )


    print("\nMessages from run 1:")
    messages = run1.get_final_state()["messages"]
    _print_messages(messages)

    print("\nMessages from run 2:")
    messages = run2.get_final_state()["messages"]
    _print_messages(messages)

    print("\nMessages from run 3:")
    messages = run3.get_final_state()["messages"]
    _print_messages(messages)
