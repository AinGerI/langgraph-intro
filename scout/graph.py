from pydantic import BaseModel
from typing import Annotated, List, Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessageChunk, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from scout.tools import query_db, generate_visualization
from scout.prompts import prompts
# 
class  ScoutState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = []
    chart_json: str = ""
# 
# state = ScoutState(
#     messages = [HumanMessage(content='hello my name is qinferl,what is  your name?')],
#     chart_json = "this is a chart json"
# )
# 
# print(state.model_dump_json(indent=2))
# 
# llm = ChatOpenAI(
#     model="gpt-4.1-mini-2025-04-14",
#     temperature=0.1,
#     openai_api_base="https://openrouter.ai/api/v1" # 关键参数：把请求指向 OpenRouter
# )
# 
# from langchain_core.tools import tool
# 
# @tool
# def raise_number_to_the_power_of(a: float, b: float) ->str:
#     """raise the number a to the  power of b"""
#     return a**b
# 
# tools = [raise_number_to_the_power_of]
# 
# llm_w_tools = llm.bind_tools(tools)
# 
# response = llm.invoke('hi,what it is the friends?')
# 
# print(response)
# 
# def assistant_node(state:ScoutState) -> ScoutState:
#     response = llm_w_tools.invoke(state.messages)
#     state.messages.append(response)
#     return state
# 
# def assistant_router(state:ScoutState) -> str:
#     last_message = state.messages[-1]
#     if not last_message.tool_calls:
#         return END
#     else:
#         return "tools"
# 
# result = assistant_node(state)
# 
# print(result.model_dump_json(indent=2))
# 
# builder = StateGraph(ScoutState)
# builder.add_node(assistant_node)
# builder.add_node(ToolNode(tools), "tools")
# builder.add_edge(START, "assistant_node")
# builder.add_conditional_edges(
#     "assistant_node",
#     assistant_router,
#     ["tools", END]
# )
# builder.add_edge("tools", "assistant_node")
# 
# demo_graph = builder.compile()  # 改名，避免和主流程冲突
# 
# from IPython.display import display, Image
# 
# # display(Image(demo_graph.get_graph(xray=True).draw_mermaid_png()))
# 
# config = {
#     "configurable":{
#         "thread_id": "1",
#     }
# }
# 
# state = ScoutState(
#     messages = [HumanMessage(content='pls answer my question in cn\
#         what is my name and what is the number as base and was the number as power')],
#     chart_json = "this is a chart json"
# )
#  
# # 以下为演示调用，主流程请用下方 Agent 类
# # result = demo_graph.invoke(
# #     input=state,
# #     config=config,
# # )
# ==================【演示代码结束】==================

# ==================【主流程：Agent 类及其用法】==================

class Agent:
    """
    Agent class for implementing Langgraph agents.

    Attributes:
        name: The name of the agent.
        tools: The tools available to the agent.
        model: The model to use for the agent.
        system_prompt: The system prompt for the agent.
        temperature: The temperature for the agent.
        openai_api_base: The base URL for the OpenAI API (default: https://openrouter.ai/api/v1)
    """
    def __init__(
            self, 
            name: str, 
            tools: List = [query_db, generate_visualization],
            model: str = "gpt-4.1-mini-2025-04-14", 
            system_prompt: str = "You are a helpful assistant.",
            temperature: float = 0.1,
            openai_api_base: str = "https://openrouter.ai/api/v1"
            ):
        self.name = name
        self.tools = tools
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.openai_api_base = openai_api_base
        
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_base=self.openai_api_base
            ).bind_tools(self.tools)
        
        self.runnable = self.build_graph()

    def build_graph(self):
        """
        Build the LangGraph application.
        """
        def scout_node(state: ScoutState) -> ScoutState:
            response = self.llm.invoke(
                [SystemMessage(content=self.system_prompt)] +
                state.messages
                )
            state.messages = state.messages + [response]
            return state
        
        def router(state: ScoutState) -> str:
            last_message = state.messages[-1]
            if not last_message.tool_calls:
                return END
            else:
                return "tools"

        builder = StateGraph(ScoutState)

        builder.add_node("chatbot", scout_node)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "chatbot")
        builder.add_conditional_edges("chatbot", router, ["tools", END])
        builder.add_edge("tools", "chatbot")

        # return builder.compile(checkpointer=MemorySaver())  # 云端/服务端环境不需要自定义checkpointer
        return builder.compile()  # 兼容LangGraph API自动持久化
    

    def inspect_graph(self):
        """
        Visualize the graph using the mermaid.ink API.
        """
        from IPython.display import display, Image

        graph = self.build_graph()
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))


    def invoke(self, message: str, **kwargs) -> str:
        """同步调用 graph，返回 LLM 回复内容。"""
        result = self.runnable.invoke(
            input = {
                "messages": [HumanMessage(content=message)]
            },
            **kwargs
        )
        return result["messages"][-1].content
    

    def stream(self, message: str, **kwargs) -> Generator[str, None, None]:
        """同步流式调用 graph，返回 LLM 回复内容或工具调用内容。"""
        for message_chunk, metadata in self.runnable.stream(
            input = {
                "messages": [HumanMessage(content=message)]
            },
            stream_mode="messages",
            **kwargs
        ):
            if isinstance(message_chunk, AIMessageChunk):
                if message_chunk.response_metadata:
                    finish_reason = message_chunk.response_metadata.get("finish_reason", "")
                    if finish_reason == "tool_calls":
                        yield "\n\n"

                if message_chunk.tool_call_chunks:
                    tool_chunk = message_chunk.tool_call_chunks[0]

                    tool_name = tool_chunk.get("name", "")
                    args = tool_chunk.get("args", "")

                    if tool_name:
                        tool_call_str = f"\n\n< TOOL CALL: {tool_name} >\n\n"

                    if args:
                        tool_call_str = args
                    yield tool_call_str
                else:
                    yield message_chunk.content
                continue

# ==========【主流程入口：定义 agent 和 graph】==========

# 定义并实例化 agent
agent = Agent(
        name="Scout",
        system_prompt=prompts.scout_system_prompt
        )
graph = agent.build_graph()

# 你可以用 agent.invoke("你的问题") 来获得回复
# 例如：
# reply = agent.invoke("请用中文介绍一下你自己")
# print(reply)
