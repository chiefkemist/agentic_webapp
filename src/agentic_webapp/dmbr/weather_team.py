#!/usr/bin/env python3
from typing import Optional, Literal, List

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from pydantic import BaseModel, Field
from pydantic_core import from_json

from agentic_webapp.dmbr.agent import Agent
from agentic_webapp.dmbr.llm import LLMModel
from agentic_webapp.dmbr.term import (
    print_user_msg,
    print_debug_msg,
    print_assistant_msg, print_error_msg,
)
from agentic_webapp.dmbr.tools import weather_icon, weather_prediction


class Prediction(BaseModel):
    humidity: float = Field(..., alias="humidity")
    temperature: float = Field(..., alias="temperature")
    description: str = Field(..., alias="description")
    icon_url: str = Field(..., alias="icon url")


class WeatherPrediction(BaseModel):
    city: str = Field(..., alias="city")
    state: Optional[str] = Field(None, alias="state")
    country: Optional[str] = Field(None, alias="country")
    predictions: List[Prediction] = Field(..., alias="predictions")


class MultiLocationWeatherPrediction(BaseModel):
    predictions_list: List[WeatherPrediction] = Field(..., alias="List of Weather Predictions")


class HumanFriendlyDescription(BaseModel):
    highly_detailed: str = Field(..., alias="highly detailed")
    concise: str = Field(..., alias="concise")
    terse: str = Field(..., alias="terse")
    large_image_url: str = Field(..., alias="large image url")
    small_image_url: str = Field(..., alias="small image url")


class WeatherPredictionDescriptions(BaseModel):
    city: str = Field(..., alias="city")
    state: Optional[str] = Field(None, alias="state")
    country: Optional[str] = Field(None, alias="country")
    descriptions: List[HumanFriendlyDescription] = Field(..., alias="descriptions")


if __name__ == "__main__":

    weather_predict = Agent(
        "weather_predictor",
        LLMModel.GPT4_Omni,
        """
        As a Weather Service Agent, I can provide weather information to users, based on their location.
        Ensure that the weather information is accurate and up-to-date and contains the proper icons urls,
        to illustrate the weather predictions. 
        """,
        [weather_icon, weather_prediction],
        output_structure=MultiLocationWeatherPrediction
    )

    user_input = "What's the weather like in Abidjan? Nairobi? Cotonou? Pretoria?"
    # user_input = "What's the weather like in Abidjan?"
    print_user_msg(user_input)
    if user_input.lower() in ["exit", "quit"]:
        print_debug_msg("Goodbye!")
    for event in weather_predict(HumanMessage(content=user_input), stream=True, debug=True):
        for value in event.values():
            print_debug_msg(value)
            print_assistant_msg(f"Assistant: {value['messages']}")
            # try:
            #     weather_predictions = from_json(value["messages"])
            #     print_assistant_msg(f"Assistant: {weather_predictions.to_json(indent=4)}")
            # except Exception as e:
            #     print_error_msg(e)
            #     print_error_msg(value['messages'])

