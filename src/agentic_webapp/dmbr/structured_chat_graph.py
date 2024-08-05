#!/usr/bin/env python3

import operator

from typing import Annotated, TypedDict, Union
from langchain_core.messages.utils import AnyMessage
from langgraph.graph import (
    add_messages,
    MessagesState,
    StateGraph,
)

from pydantic import BaseModel, Field

from agentic_webapp.dmbr.llm import get_llm, LLMModel
from agentic_webapp.dmbr.term import (
    print_user_msg,
    print_assistant_msg,
    print_debug_msg,
)


class SampleCode(BaseModel):
    markdown: str = Field(
        ..., title="Markdown", description="The markdown explanation of the code"
    )
    code: str = Field(
        ...,
        title="Code",
        description="The code snippet which implements the functionality",
    )


class MsgsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    code_sample: SampleCode


llm = get_llm(LLMModel.GPT4_Omni_mini)
structured_llm = llm.with_structured_output(SampleCode)


def chatbot(state: MsgsState):
    return dict(messages=llm.invoke(state["messages"]))


def sample_code_parser(state: MsgsState):
    print_debug_msg(state["messages"])
    return dict(code_sample=structured_llm.invoke(state["messages"]))


structured_chat_flow_builder = StateGraph(MsgsState)
structured_chat_flow_builder.add_node("chatbot", chatbot)
structured_chat_flow_builder.add_node("sample_code_parser", sample_code_parser)
structured_chat_flow_builder.set_entry_point("chatbot")
structured_chat_flow_builder.add_edge("chatbot", "sample_code_parser")
structured_chat_flow_builder.set_finish_point("sample_code_parser")


structured_chat_flow = structured_chat_flow_builder.compile()

if __name__ == "__main__":
    while True:
        user_input = input("Student: ")
        print_user_msg(user_input)
        if user_input.lower() in ["exit", "quit"]:
            print_debug_msg("Goodbye!")
            break
        for event in structured_chat_flow.stream(dict(messages=("user", user_input))):
            for value in event.values():
                print_debug_msg(value)
                if "code_sample" in value:
                    print_assistant_msg(f"Professor:\n{value['code_sample'].markdown}")
                    print_assistant_msg(f"Professor:\n{value['code_sample'].code}")
                else:
                    print_assistant_msg(f"Professor:\n{value['messages'].content}")
