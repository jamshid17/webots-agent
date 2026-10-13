import operator
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.messages import AnyMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


async def make_graph():
    # 1. Initialize MCP Client
    client = MultiServerMCPClient(
        {
            "webots_mcp": {
                "command": "/Users/jamshid/.pyenv/shims/python",
                "args": [
                    "/Users/jamshid/Documents/uni/project1/webots-mcp/webots_mcp_server.py"
                ],
                "transport": "stdio",
            }
        }
    )

    # 2. Retrieve MCP tools asynchronously
    tools = await client.get_tools()

    # 3. Initialize model & bind tools

    model = ChatOpenAI(
        model="Qwen/Qwen3.8-27B-NVFP4",  # Must match the model name registered on your server
        api_key=os.environ["QWEN_API_KEY"],
        base_url=os.environ["QWEN_BASE_URL"],
        temperature=0.7,
    )
    model_with_tools = model.bind_tools(tools)

    # 4. Define LLM node
    def llm_call(state: MessagesState):
        return {
            "messages": [
                model_with_tools.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant capable of controlling and interacting with the Webots simulation via MCP tools."
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # 5. Conditional edge router
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # 6. Build workflow
    agent_builder = StateGraph(MessagesState)

    agent_builder.add_node("llm_call", llm_call)
    # Built-in ToolNode automatically manages tool execution & ToolMessage outputs
    agent_builder.add_node("tools", ToolNode(tools))

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tools", END])
    agent_builder.add_edge("tools", "llm_call")

    return agent_builder.compile()
