# UdaPlayOO

A conversational AI agent for discovering and recommending video games. The agent uses retrieval-augmented generation (RAG), web search, and long-term memory to answer questions about games based on a local knowledge base and real-time web results.

## Features

- Game information retrieval from a local `games.json` knowledge base
- Web search integration via Tavily for up-to-date game information
- Long-term memory: the agent can store and recall facts across sessions
- RAG pipeline with vector store for semantic search over game data
- Modular agent architecture with tool-calling support

## Project Structure

```
main.py          — Entry point; defines the agent setup and conversation loop
tools.py         — Tool definitions (retrieval, web search, memory)
games.json       — Local game knowledge base
lib/             — Core agent framework (see attribution below)
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables by creating a `config.env` file:
   ```
   OPENAI_API_KEY=<your-openai-api-key>
   TAVILY_API_KEY=<your-tavily-api-key>
   ```

4. Run the agent:
   ```bash
   python main.py
   ```

## Attribution

The code under the `lib/` directory — including the agent framework, LLM abstraction, message handling, memory management, RAG pipeline, state machine, tooling utilities, and vector database integration — is based on course material from the Udacity training:

**Building AI Agents** — [https://learn.udacity.com/cd14524](https://learn.udacity.com/cd14524)

## License

See [LICENSE](LICENSE) for details.
