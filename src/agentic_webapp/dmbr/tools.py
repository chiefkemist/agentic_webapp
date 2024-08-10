#!/usr/bin/env python3

import operator
import os
from typing import Any, Literal, Optional

import httpx
from langchain_core.tools import tool

from agentic_webapp.dmbr.term import print_debug_msg


@tool("weather_icon")
def weather_icon(icon: str, size: Literal["2", "4"]) -> str:
    """
    Weather Icon: Get the icon for the weather
    """
    return f"https://openweathermap.org/img/wn/{icon}@{size}x.png"


@tool("weather_prediction")
def weather_prediction(city: str, state: Optional[str], country: Optional[str]) -> str:
    """
    Weather Prediction: Get the prediction for the weather
    """
    app_id = os.getenv("OPENWEATHERMAP_API_KEY")
    if state is None and country is None:
        prediction = httpx.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&APPID={app_id}"
        ).json()
        print_debug_msg(f"Weather Prediction for {city} is: {prediction}")
    if state is not None and country is None:
        prediction = httpx.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city},{state}&APPID={app_id}"
        ).json()
        print_debug_msg(f"Weather Prediction for {city} {state} is: {prediction}")
    if state is None and country is not None:
        prediction = httpx.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city},{country}&APPID={app_id}"
        ).json()
        print_debug_msg(f"Weather Prediction for {city} {country} is: {prediction}")
    if state is not None and country is not None:
        prediction = httpx.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city},{state},{country}&APPID={app_id}"
        ).json()
        print_debug_msg(
            f"Weather Prediction for {city} {state} {country} is: {prediction}"
        )
    return prediction


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
