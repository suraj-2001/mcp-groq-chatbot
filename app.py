import asyncio
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient


async def run_chat():
    """Interactive chatbot using MCPAgent with built-in conversation memory."""
    # Load GROQ_API_KEY from the .env file
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")

    # 1. MCP client: starts/connects to the MCP servers listed in the config file
    client = MCPClient.from_config_file("browser_mcp.json")

    # 2. LLM: must support tool calling (check console.groq.com/docs/models)
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    # 3. Agent: the LLM decides which MCP tools to call, step by step
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,  # keeps conversation history between turns
    )

    print("MCP chatbot ready.")
    print("Type 'exit' or 'quit' to leave, 'clear' to reset the conversation.")

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            print("\nAssistant: ", end="", flush=True)
            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"\nError: {e}")
    finally:
        # Shut down MCP server processes cleanly
        if client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(run_chat())