import os
import re

from typing import Type, Optional, Literal
from pydantic import BaseModel

from openai import OpenAI
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=os.environ["OPENAI_ORGANIZATION_ID"],
)


OPENAI_MODELS_WITHOUT_TEMPERATURE = ["o3-mini", "o1", "o1-mini"]


def query_openai_model(
    messages: list[ChatCompletionMessageParam],
    model: str = "gpt-3.5-turbo",
    max_output_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 120,
    response_type: Literal["text", "json_object"] = "text",
) -> dict:
    response_format = {"type": response_type}
    has_no_temperature = any(
        model_name in model for model_name in OPENAI_MODELS_WITHOUT_TEMPERATURE
    )
    response = client.chat.completions.create(
        model=model,
        temperature=NOT_GIVEN if has_no_temperature else temperature,
        messages=messages,
        max_completion_tokens=max_output_tokens,
        timeout=timeout,
        response_format=response_format,  # type: ignore[call-overload]
    )
    return dict(response.choices[0].message)


def query_openai_model_structured_outputs(
    messages: list[ChatCompletionMessageParam],
    output_class: Type[BaseModel],
    model: str = "gpt-4o-2024-08-06",
    max_completion_tokens: int = 5000,
    temperature: float = 0.0,
    timeout: int = 120,
) -> Optional[BaseModel]:
    has_no_temperature = any(
        model_name in model for model_name in OPENAI_MODELS_WITHOUT_TEMPERATURE
    )
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=output_class,
        max_completion_tokens=max_completion_tokens,
        temperature=NOT_GIVEN if has_no_temperature else temperature,
        timeout=timeout,
    )
    return completion.choices[0].message.parsed


def replace_markdown_links_with_text(sentence: str, replacement: str) -> str:
    return re.sub(r"\[((?:\[)?([^]]+)(?:\])?)\]\(([^)]+)\)", replacement, sentence)
