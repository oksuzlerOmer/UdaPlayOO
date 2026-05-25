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

    # db = VectorStoreManager(os.getenv("OPENAI_API_KEY"))

    # rag_llm = LLM(model="gpt-4o-mini", temperature=0.3)

    # vector_store_manager = VectorStoreManager(os.getenv("OPENAI_API_KEY"))
    # games_data_vector_store = vector_store_manager.create_store(store_name="games_data", force=True)
    # games_data_vector_store.add(Document(content=get_games(num_games=10, top=True)))

    # games_rag = RAG(llm=rag_llm, vector_store=games_data_vector_store)

    # run=games_rag.invoke("What are the top 3 games?")
    # print(run.get_final_state()["answer"])

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
            "1.1 if a game is found, call evaluate_retrieval to check if the retrieved game information is relevant and sufficient to answer the user's query. If the evaluation score is above 70%, you can use this information to answer the query without calling any more tools.\n"
            "2. Only if retrieve_game returns no result or insufficient information, call the long-term memory search tool to check previously stored knowledge.\n"
            "3. Only if the long-term memory search also returns no useful result, call game_web_search to find the information online.\n"
            "4. After using game_web_search, you MUST always call the long-term memory registration tool to persist the retrieved information for future use.\n"
            "Never skip steps or jump ahead. Always incorporate all retrieved information into your final response. Be concise and informative."
        ),
    )

    # run1 = my_agent.invoke(
    #     query="give me score and platform info about Evolve", session_id="session_1"
    # )

    run2 = my_agent.invoke(
        query="give me score and platform info about Metroid", session_id="session_2"
    )

    # run3 = my_agent.invoke(
    #     query="give me score and platform info about Evolve", session_id="session_2"
    # )

    # run4 = my_agent.invoke(
    #     query="give me score and platform info about Evolve", session_id="session_1"
    # )

    print("\nMessages from run 1:")
    # messages = run1.get_final_state()["messages"]
    # _print_messages(messages)
