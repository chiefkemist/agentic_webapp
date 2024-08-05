#!/usr/bin/env python3

import operator
from typing import Annotated, Any, Iterator, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage, tool_call
from langchain_core.messages.utils import AnyMessage
from langchain_core.tools import tool
from langgraph import graph
from langgraph.constants import END
from langgraph.graph import add_messages
from agentic_webapp.dmbr.llm import get_llm, LLMModel
from agentic_webapp.dmbr.term import (
    print_user_msg,
    print_assistant_msg,
    print_debug_msg,
    print_error_msg,
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Agent:
    def __init__(self, model: LLMModel, system="", tools=[]):
        print_debug_msg(f"Initializing agent with model {model}")
        self.system = system
        llm = get_llm(model)
        graph_builder = graph.StateGraph(AgentState)
        graph_builder.add_node("llm", self.call_llm)
        graph_builder.add_node("action", self.act)
        graph_builder.add_conditional_edges(
            "llm", self.should_act, {True: "action", False: END}
        )
        graph_builder.add_edge("action", "llm")
        graph_builder.set_entry_point("llm")
        self.graph = graph_builder.compile()
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm.bind_tools(tools)

    def should_act(self, state: AgentState):
        print_debug_msg(f"Checking if action exists in {state['messages'][-1]}")
        result = state["messages"][-1]
        tool_calls_count = len(result.tool_calls)
        return tool_calls_count > 0

    def call_llm(self, state: AgentState):
        print_debug_msg(f"Calling LLM with state {state}")
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.llm.invoke(messages)
        return {"messages": [message]}

    def act(self, state: AgentState):
        print_debug_msg(f"Taking action on message {state['messages'][-1]}")
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print_debug_msg(f"Calling: {t}")
            if not t["name"] in self.tools:
                print_error_msg(f"Tool {t['name']} not found")
                result = "Tool not found, please try again"
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print_debug_msg("Back to model after action")
        return {"messages": results}

    def __call__(self, message: HumanMessage) -> Iterator[dict]:
        return self.graph.stream(dict(messages=message))


if __name__ == "__main__":
    system = """
    As a basic arithmetic agent, I can perform the following operations:
    - Addition
    - Subtraction
    - Multiplication
    - Division
    Let me know if you need help with any of these operations.
    """

    @tool("add")
    def add(a: Any, b: Any) -> Any:
        """
        Addition: Add two numbers together
        """
        return operator.add(a, b)

    @tool("sub")
    def sub(a: Any, b: Any) -> Any:
        """
        Substraction: Substract one number from the other
        """
        return operator.sub(a, b)

    @tool("mul")
    def mul(a: Any, b: Any) -> Any:
        """
        Multiplication: Multiply two numbers
        """
        return operator.mul(a, b)

    @tool("truediv")
    def truediv(a: Any, b: Any) -> Any:
        """
        Division: Divide one number by the other
        """
        return operator.truediv(a, b)

    tools = [
        add,
        sub,
        mul,
        truediv,
    ]

    agent_calculate = Agent(LLMModel.GPT4_Omni, system, tools)

    while True:
        # Example query: Ms Adjo receives a stock of 50000 pineapples a month, and is able to sell 4/5 of it. What is her monthly gross revenue given that pineapples go for $5 a piece? What are her monthly losses, making sure to calculate her losses based on the unsold pineapples. What is the ratio of her revenue to her losses? Is she profitable knowing that she buys her stock of pineapples at $2 a piece and has a total of $80000 of operating expenses? Respond with plain language.
        user_input = input("User: ")
        print_user_msg(user_input)
        if user_input.lower() in ["exit", "quit"]:
            print_debug_msg("Goodbye!")
            break
        for event in agent_calculate(HumanMessage(content=user_input)):
            for value in event.values():
                print_debug_msg(value)
                print_assistant_msg(f"Assistant: {value['messages']}")
