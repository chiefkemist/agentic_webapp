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
    print_assistant_msg,
    print_error_msg,
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
    predictions_list: List[WeatherPrediction] = Field(
        ..., alias="List of Weather Predictions"
    )


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

    @tool("predict_weather", return_direct=True)
    def predict_weather(data: str) -> str:
        """
        Weather Prediction: Get the prediction for the weather
        """
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
        for event in weather_predict(
            HumanMessage(content=f"What's the weather like for {data}?"),
            stream=True,
            debug=True,
        ):
            for value in event.values():
                try:
                    weather_predictions = value["messages"]
                    weather_predictions_obj = from_json(weather_predictions)
                    print_assistant_msg(
                        f"Assistant: {weather_predictions_obj.to_json(indent=4)}"
                    )
                    return f"The JSON data of the weather for {data} predictions is: {weather_predictions}"
                except Exception as e:
                    print_error_msg(e)

    @tool("describe_weather", return_direct=True)
    def describe_weather(data: str) -> str:
        """
        Weather Description: Describe the weather from the provided prediction data
        """
        weather_describe = Agent(
            "weather_describer",
            LLMModel.GPT4_Omni,
            """
            As a Weather Service Agent, I can provide human-friendly descriptions of the weather predictions to users, based on their location.
            Ensure that the descriptions are accurate and up-to-date and contain the proper image urls to illustrate the weather predictions.
            """,
            output_structure=WeatherPredictionDescriptions,
        )
        for event in weather_describe(
            HumanMessage(
                content=f"Provide a human-friendly description of the weather from the raw data: {data}"
            ),
            stream=True,
            debug=True,
        ):
            for value in event.values():
                try:
                    weather_descriptions = value["messages"]
                    weather_descriptions_obj = from_json(weather_descriptions)
                    print_assistant_msg(
                        f"Assistant: {weather_descriptions_obj.to_json(indent=4)}"
                    )
                    return f"The JSON data containing the human-friendly descriptions of the weather for {data} is: {weather_descriptions}"
                except Exception as e:
                    print_error_msg(e)

    weather_director = Agent(
        "weather_director",
        LLMModel.GPT4_Omni,
        """
        As a Weather Service Agent, I can provide weather information to users, based on their location.
        Ensure that the weather information is accurate and up-to-date and contains the proper image urls to illustrate the weather predictions.
        """,
        tools=[predict_weather, describe_weather],
    )

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

    # while True:
    #     user_input = input("User: ")
    #     print_user_msg(user_input)
    #     if user_input.lower() in ["exit", "quit"]:
    #         print_debug_msg("Goodbye!")
    #         break
    #     for event in weather_predict(HumanMessage(content=user_input), stream=True, debug=True):
    #         for value in event.values():
    #             print_debug_msg(value)
    #             print_assistant_msg(f"Assistant: {value['messages']}")
    #             try:
    #                 weather_predictions = MultiLocationWeatherPrediction.model_validate_json(value["messages"])
    #                 print_assistant_msg(f"Assistant: {weather_predictions.predictions.to_json(indent=4)}")
    #             except Exception as e:
    #                 pass

    # user_input = "What's the weather like in Abidjan? Nairobi? Cotonou? Pretoria?"
    user_input = "What's the weather like in Abidjan?"
    print_user_msg(user_input)
    if user_input.lower() in ["exit", "quit"]:
        print_debug_msg("Goodbye!")
    for event in weather_director(
        HumanMessage(content=user_input), stream=True, debug=True
    ):
        for value in event.values():
            print_debug_msg(value)
            print_assistant_msg(f"Assistant: {value['messages']}")
            # try:
            #     weather_predictions = from_json(value["messages"])
            #     print_assistant_msg(f"Assistant: {weather_predictions.to_json(indent=4)}")
            # except Exception as e:
            #     print_error_msg(e)
            #     print_error_msg(value['messages'])
