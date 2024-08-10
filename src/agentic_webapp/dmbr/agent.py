#!/usr/bin/env python3

import operator
from typing import Annotated, Any, Iterator, TypedDict, Dict
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
    def __init__(
        self,
        name,
        model: LLMModel,
        system="",
        tools=[],
        output_structure=None,
        delegates: Dict[str, Any] = None,
    ):
        print_debug_msg(f"Initializing agent with model {model}")
        self.system = system
        llm = get_llm(model)
        graph_builder = graph.StateGraph(AgentState)
        graph_builder.add_node(name, self.call_llm)
        graph_builder.add_node("action", self.act)
        # if delegates:
        #     for delegate_name, delegate in delegates.items():
        #         graph_builder.add_node(delegate_name, delegate)
        #     for delegate_name in delegates.keys():
        #         graph_builder.add_edge(delegate_name, name)
        #         graph_builder.add_edge(name, delegate_name)
        if output_structure:
            graph_builder.add_node("output_parser", self.output_parser)
            graph_builder.add_conditional_edges(
                name, self.should_act, {True: "action", False: "output_parser"}
            )
            graph_builder.set_finish_point("output_parser")
        else:
            graph_builder.add_conditional_edges(
                name, self.should_act, {True: "action", False: END}
            )
        graph_builder.add_edge("action", name)
        graph_builder.set_entry_point(name)
        self.graph = graph_builder.compile()
        if len(tools) > 0:
            self.tools = {tool.name: tool for tool in tools}
            self.llm = llm.bind_tools(tools)
        else:
            self.llm = llm
        self.output_structure = output_structure

    def output_parser(self, state: AgentState):
        print_debug_msg(f"Output parser with state {state['messages']}")
        llm_with_output_structure = self.llm.with_structured_output(
            self.output_structure
        )
        structured_output = llm_with_output_structure.invoke(state["messages"])
        return {"messages": structured_output.json()}

    def should_act(self, state: AgentState):
        print_debug_msg(f"Checking if action exists in {state['messages']}")
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

    def __call__(
        self, message: HumanMessage, stream=False, debug=False
    ) -> Iterator[dict]:
        if stream:
            results = self.graph.stream(dict(messages=message), debug=debug)
            return results
        else:
            results = self.graph.invoke(dict(messages=message), debug=debug)
            return results
