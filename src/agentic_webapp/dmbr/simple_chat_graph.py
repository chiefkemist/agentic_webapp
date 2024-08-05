#!/usr/bin/env python3

from langgraph.graph import (
    MessagesState,
    StateGraph,
)
from langchain_anthropic import ChatAnthropic

from agentic_webapp.dmbr.llm import get_llm, LLMModel
from agentic_webapp.dmbr.term import (
    print_user_msg,
    print_assistant_msg,
    print_debug_msg,
)


llm = get_llm(LLMModel.GPT4_Omni_mini)


def chatbot(state: MessagesState):
    return dict(messages=llm.invoke(state["messages"]))


simple_chat_flow_builder = StateGraph(MessagesState)

simple_chat_flow_builder.add_node("chatbot", chatbot)
simple_chat_flow_builder.set_entry_point("chatbot")
simple_chat_flow_builder.set_finish_point("chatbot")


simple_chat_flow = simple_chat_flow_builder.compile()

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        print_user_msg(user_input)
        if user_input.lower() in ["exit", "quit"]:
            print_debug_msg("Goodbye!")
            break
        for event in simple_chat_flow.stream(dict(messages=("user", user_input))):
            for value in event.values():
                print_assistant_msg(f"Assistant: {value['messages'].content}")
