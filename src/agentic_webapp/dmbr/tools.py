#!/usr/bin/env python3

import operator
from typing import Any
from langchain_core.tools import tool


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
