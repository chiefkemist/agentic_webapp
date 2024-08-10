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
from pydantic_core import from_json
from starlette.responses import StreamingResponse

from langchain_core.messages import HumanMessage

from agentic_webapp.dmbr.agent import Agent
from agentic_webapp.dmbr.tools import weather_prediction, weather_icon
from agentic_webapp.dmbr.weather_team import MultiLocationWeatherPrediction

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


from agentic_webapp.dmbr.llm import get_llm, LLMModel
from agentic_webapp.dmbr.term import (
    print_user_msg,
    print_assistant_msg, print_error_msg,
)

weather_predict = Agent(
    "weather_predictor",
    LLMModel.GPT4_Omni,
    """
        As a Weather Service Agent, I can provide weather information to users, based on their location.
        Ensure that the weather information is accurate and up-to-date and contains the proper icons urls,
        to illustrate the weather predictions. 
        """,
    [weather_icon, weather_prediction],
    output_structure=MultiLocationWeatherPrediction,
)


async def weather_chat(user_input: str):
    print_user_msg(user_input)
    for event in weather_predict(HumanMessage(content=user_input), stream=True, debug=True):
        for value in event.values():
            content = value["messages"]
            print_assistant_msg(f"Assistant: {content}")
            try:
                weather_predictions = value["messages"]
                weather_predictions_obj = from_json(weather_predictions)
                print_assistant_msg(
                    f"Assistant: {weather_predictions_obj.to_json(indent=4)}"
                )
                yield weather_predictions
            except Exception as e:
                print_error_msg(e)
            # if type(content) == list:
            #     for c in content:
            #         if 'content' in c:
            #             yield c.content
            #         else:
            #             yield c
            # elif 'content' in content:
            #     yield content.content
            # else:
            #     yield content


def render_sse_html_chunk(event: str, id: str, chunk: str, hx_swap_oob="true") -> bytes:
    return f"""
event: {event}
data: {to_xml(Div(chunk, id=id, hx_swap_oob=hx_swap_oob))}\n\n
""".encode("utf-8")


@route("/chatstream")
def get(request):
    prompt = request.query_params["prompt"]

    async def chat_iter():
        async for chat in weather_chat(prompt):
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
    return Titled("Weather Chat"), chat_log, Main(add, cls="container")


if __name__ == "__main__":
    serve(port=8000, reload=True)
