#!/usr/bin/env python3

from fasthtml import P, Link, Script, Titled
from fasthtml.fastapp import fast_app, serve


app, route = fast_app(
    debug=True,
    live=True,
    hdrs=(
        Link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.pumpkin.min.css",
            type="text/css",
        ),
        Script(src="https://unpkg.com/htmx.org@1.9.12/dist/ext/sse.js"),
    ),
)


@route("/")
def get():
    return Titled("Home", P("Hello, world!"))


if __name__ == "__main__":
    serve()
