#!/usr/bin/env python3
import asyncio

from fasthtml import (
    Link,
    Script,
    Titled,
    Div,
    Hr,
    B,
    Br,
    Form,
    Input,
    Group,
    Button,
    Main,
)
from fasthtml.common import to_xml
from fasthtml.fastapp import fast_app, serve
from starlette.responses import StreamingResponse

app, route = fast_app(
    debug=True,
    live=True,
    hdrs=(
        Link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.pumpkin.min.css",
            type="text/css",
        ),
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
    ),
)

from langgraph.graph import (
    MessagesState,
    StateGraph,
)

from agentic_webapp.dmbr.llm import get_llm, LLMModel
from agentic_webapp.dmbr.term import (
    print_user_msg,
    print_assistant_msg,
)

llm = get_llm(LLMModel.GPT4_Omni_mini)


def chatbot(state: MessagesState):
    return dict(messages=llm.invoke(state["messages"]))


simple_chat_flow_builder = StateGraph(MessagesState)

simple_chat_flow_builder.add_node("chatbot", chatbot)
simple_chat_flow_builder.set_entry_point("chatbot")
simple_chat_flow_builder.set_finish_point("chatbot")

simple_chat_flow = simple_chat_flow_builder.compile()


async def simple_chat(user_input: str):
    print_user_msg(user_input)
    for event in simple_chat_flow.stream(dict(messages=("user", user_input))):
        for value in event.values():
            content = value["messages"].content
            print_assistant_msg(f"Assistant: {content}")
            yield content


def render_sse_html_chunk(event: str, id: str, chunk: str, hx_swap_oob="true") -> bytes:
    return f"""
event: {event}
data: {to_xml(Div(chunk, id=id, hx_swap_oob=hx_swap_oob))}\n\n
""".encode("utf-8")


@route("/chatstream")
def get(request):
    prompt = request.query_params["prompt"]

    async def chat_iter():
        async for chat in simple_chat(prompt):
            await asyncio.sleep(1)
            chat_status_chunk = render_sse_html_chunk("Status", "Status", "Sending...")
            yield chat_status_chunk
            await asyncio.sleep(1)
            chunk = render_sse_html_chunk("Chat", "Chat", chat, hx_swap_oob="beforeend")
            yield chunk
        chat_status_chunk = render_sse_html_chunk("Status", "Status", "Answered")
        yield chat_status_chunk
        terminating_chunk = render_sse_html_chunk("Terminate", "Terminate", "")
        yield terminating_chunk

    return StreamingResponse(
        chat_iter(),
        media_type="text/event-stream",
    )


@route("/query")
def post(prompt: str):
    return Main(
        Div(
            id="Terminate",
            hx_ext="sse",
            sse_connect=f"/chatstream?prompt={prompt}",
            sse_swap="Terminate,Status,Chat",
        ),
        B(id="Status", sse_swap="Status"),
        Br(),
        Div(id="Chat", sse_swap="Chat"),
        cls="container",
    )


@route("/")
def get():
    chat_log = Div(id="chat-log")
    inp = Input(id="new-prompt", name="prompt", placeholder="Enter a prompt")
    add = Form(
        Group(inp, Button("Query")),
        hx_post="/query",
        target_id="chat-log",
        hx_swap="afterbegin",
    )
    return Titled("Simple Web Chat"), chat_log, Main(add, cls="container")


if __name__ == "__main__":
    serve(port=8000, reload=True)
