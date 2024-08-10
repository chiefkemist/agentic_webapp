#!/usr/bin/env python3
from typing import Optional, Literal, List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agentic_webapp.dmbr.agent import Agent
from agentic_webapp.dmbr.llm import LLMModel
from agentic_webapp.dmbr.term import (
    print_user_msg,
    print_debug_msg,
    print_assistant_msg,
)
from agentic_webapp.dmbr.tools import weather_icon, weather_prediction


class WeatherIcon(BaseModel):
    icon_name: str = Field(..., alias="icon name")
    size: Literal["2", "4"] = Field(..., alias="size")
    icon_url: str = Field(..., alias="icon url")


class Prediction(BaseModel):
    humidity: float = Field(..., alias="humidity")
    temperature: float = Field(..., alias="temperature")
    description: str = Field(..., alias="description")
    icon: WeatherIcon = Field(..., alias="icon")


class WeatherPrediction(BaseModel):
    city: str = Field(..., alias="city")
    state: Optional[str] = Field(None, alias="state")
    country: Optional[str] = Field(None, alias="country")
    predictions: List[Prediction] = Field(..., alias="predictions")


class MultiLocationWeatherPrediction(BaseModel):
    predictions: List[WeatherPrediction] = Field(..., alias="predictions")


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
        Ensure that the weather information is accurate and up-to-date and contains the proper image urls to illustrate the weather predictions. 
        """,
        [weather_icon, weather_prediction],
        output_structure=MultiLocationWeatherPrediction
    )

    # weather_describe = Agent(
    #     "weather_describer",
    #     LLMModel.GPT4_Omni,
    #     """
    #     As a Weather Service Agent, I can provide human-friendly descriptions of the weather predictions to users, based on their location.
    #     Ensure that the descriptions are accurate and up-to-date and contain the proper image urls to illustrate the weather predictions.
    #     """,
    #     output_structure=WeatherPredictionDescriptions
    # )

    # weather_director = Agent(
    #     "weather_director",
    #     LLMModel.GPT4_Omni,
    #     """
    #     As a Weather Service Agent, I can provide weather information to users, based on their location.
    #     Ensure that the weather information is accurate and up-to-date and contains the proper image urls to illustrate the weather predictions.
    #     To assist me in providing the weather information, I have two delegates:
    #     - weather_predictor: To predict the weather
    #     - weather_describer: To describe the weather predictions
    #     """,
    #     # delegates={
    #     #     "weather_predictor": weather_predict,
    #     #     "weather_describer": weather_describe,
    #     # }
    # )

    while True:
        # user_input = input("User: ")
        user_input = "What's the weather like in Abidjan? Nairobi? Cotonou? Pretoria?"
        print_user_msg(user_input)
        if user_input.lower() in ["exit", "quit"]:
            print_debug_msg("Goodbye!")
            break
        for event in weather_predict(HumanMessage(content=user_input), stream=True, debug=True):
            for value in event.values():
                print_debug_msg(value)
                print_assistant_msg(f"Assistant: {value['messages']}")
                try:
                    weather_predictions = MultiLocationWeatherPrediction.model_validate_json(value["messages"])
                    print_assistant_msg(f"Assistant: {weather_predictions.predictions.to_json(indent=4)}")
                except Exception as e:
                    pass
