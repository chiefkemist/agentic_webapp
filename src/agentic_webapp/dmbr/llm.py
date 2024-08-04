#!/usr/bin/env python3


from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


class LLMModel(str, Enum):
    Claude3_Opus = "claude-3-opus-20240229"
    Claude35_Sonnet = "claude-3-5-sonnet-20240620"
    Claude3_Haiku = "claude-3-haiku-20240307"
    GPT4_Omni = "gpt-4o"
    GPT4_Omni_mini = "gpt-4o-mini"
    GPT35_Turbo = "gpt-3.5-turbo"
    LLAMA31_70b = "llama-3.1-70b-versatile"
    LLAMA31_8b = "llama-3.1-8b-instant"
    LLAMA3_70b = "llama3-70b-8192"
    LLAMA3_8b = "llama3-8b-8192"


def get_llm(model_name: LLMModel):
    llm = {
        LLMModel.Claude3_Opus: ChatAnthropic(model_name=LLMModel.Claude3_Opus),
        LLMModel.Claude35_Sonnet: ChatAnthropic(model_name=LLMModel.Claude35_Sonnet),
        LLMModel.Claude3_Haiku: ChatAnthropic(model_name=LLMModel.Claude3_Haiku),
        LLMModel.GPT4_Omni: ChatOpenAI(model_name=LLMModel.GPT4_Omni),
        LLMModel.GPT35_Turbo: ChatOpenAI(model_name=LLMModel.GPT35_Turbo),
        LLMModel.LLAMA31_70b: ChatGroq(model_name=LLMModel.LLAMA31_70b),
        LLMModel.LLAMA31_8b: ChatGroq(model_name=LLMModel.LLAMA31_8b),
        LLMModel.LLAMA3_70b: ChatGroq(model_name=LLMModel.LLAMA3_70b),
        LLMModel.LLAMA3_8b: ChatGroq(model_name=LLMModel.LLAMA3_8b),
    }.get(model_name, None)

    if llm is None:
        raise ValueError(f"Model {model_name} not found")

    return llm
