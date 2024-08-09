#!/usr/bin/env python3
import asyncio

import httpx
from fasthtml import P, Link, Script, Titled, Div, H1, Hr, B, Br
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


async def gen_dog_breeds():
    async with httpx.AsyncClient() as client:
        breeds = (await client.get("https://dog.ceo/api/breeds/list/all")).json()
        for breed in breeds["message"].keys():
            print(f"Yielding {breed}")
            yield breed


def render_sse_html_chunk(event: str, id: str, chunk: str) -> bytes:
    return f"""
event: {event}
data: {to_xml(Div(chunk, id=id, hx_swap_oob='true'))}\n\n
""".encode("utf-8")


@route("/dogstream")
def get():
    async def dogbreeds_iter():
        async for breed in gen_dog_breeds():
            await asyncio.sleep(0.2)
            breed_status_chunk = render_sse_html_chunk(
                "DogBreedNoMass", "DogBreedNoMass", "More doggo senior :-)"
            )
            yield breed_status_chunk
            await asyncio.sleep(0.2)
            chunk = render_sse_html_chunk("DogBreed", "DogBreed", breed)
            yield chunk
        breed_status_chunk = render_sse_html_chunk(
            "DogBreedNoMass", "DogBreedNoMass", "No more doggo senior :-("
        )
        yield breed_status_chunk

    return StreamingResponse(
        dogbreeds_iter(),
        media_type="text/event-stream",
    )


@route("/doggo")
def get():
    return Titled(
        "Dog Breeds as Server Sent Events",
        Hr(),
        Div(
            id="doggo-sse-listener",
            hx_ext="sse",
            sse_connect="/dogstream",
            sse_swap="Terminate,DogBreedNoMass,DogBreed",
        ),
        B(
            Div(id="DogBreedNoMass"),
            Br(),
            Div(id="DogBreed"),
        ),
    )


@route("/")
def get():
    return Titled("Home", P("Hello, world!"))


if __name__ == "__main__":
    serve(port=8000, reload=True)
