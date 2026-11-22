"""Tests for the Claude version based special cases.

Newer Claude models reject some request shapes that older ones accept: a conversation ending with an
assistant message (from 4.6), temperature / top_p (from 4.7) and thinking with budget_tokens (from 4.7).
The gateway derives the Claude version from the model id instead of keeping lists of model names.
"""

import logging
from unittest import mock

import pytest
from fastapi import HTTPException

# api.models.bedrock creates boto3 clients, lists models and loads the tiktoken
# encoding when imported. Stub those so the import makes no network calls.
with mock.patch("boto3.client"), mock.patch("tiktoken.get_encoding"):
    from api.models import bedrock as bedrock_module

from api.models.bedrock import BedrockModel, claude_model_version
from api.schema import ChatRequest

SONNET_5_5 = "global.anthropic.claude-sonnet-5-5"
OPUS_5_5 = "au.anthropic.claude-opus-5-5"
OPUS_4_6 = "anthropic.claude-opus-4-6-v1"
SONNET_4_5 = "anthropic.claude-sonnet-4-5-20250929-v1:0"
SONNET_3_7 = "anthropic.claude-3-7-sonnet-20250219-v1:0"
NEW_MODELS = [SONNET_5_5, OPUS_5_5]

CONTINUE_PROMPT = {"role": "user", "content": [{"text": "Please continue your response from where you left off."}]}


@pytest.fixture(autouse=True)
def no_prompt_caching(monkeypatch):
    """Keep cachePoint blocks out of the request so only the model handling is exercised."""
    monkeypatch.setattr(bedrock_module, "ENABLE_PROMPT_CACHING", False)


def parse(model, messages=None, **kwargs) -> dict:
    messages = messages or [{"role": "user", "content": "Hello"}]
    return BedrockModel()._parse_request(ChatRequest(model=model, messages=messages, **kwargs))


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("global.anthropic.claude-sonnet-5-5", ("sonnet", (5, 5))),
        ("au.anthropic.claude-opus-5-5", ("opus", (5, 5))),
        ("apac.anthropic.claude-sonnet-4-5-20250929-v1:0", ("sonnet", (4, 5))),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", ("haiku", (4, 5))),
        ("eu.anthropic.claude-opus-4-6-v1", ("opus", (4, 6))),
        ("anthropic.claude-opus-4-6-v1:0", ("opus", (4, 6))),
        ("anthropic.claude-sonnet-4-6", ("sonnet", (4, 6))),
        ("anthropic.claude-opus-4-1-20250805-v1:0", ("opus", (4, 1))),
        ("anthropic.claude-sonnet-4-20250514-v1:0", ("sonnet", (4, 0))),
        ("anthropic.claude-sonnet-5", ("sonnet", (5, 0))),
        ("global.anthropic.claude-sonnet-5-v1:0", ("sonnet", (5, 0))),
        ("claude-opus-4-6", ("opus", (4, 6))),
        ("claude-sonnet-5-5", ("sonnet", (5, 5))),
        ("claude-haiku-4-5", ("haiku", (4, 5))),
        ("Global.Anthropic.Claude-Sonnet-5-5", ("sonnet", (5, 5))),
        # Older ids put the version before the family; they get no version based special case.
        ("anthropic.claude-3-7-sonnet-20250219-v1:0", None),
        ("us.anthropic.claude-3-5-sonnet-20240620-v1:0", None),
        ("anthropic.claude-v2:1", None),
        ("anthropic.claude-instant-v1", None),
        ("amazon.nova-pro-v1:0", None),
        ("deepseek.v3-v1:0", None),
        ("meta.llama3-1-70b-instruct-v1:0", None),
    ],
)
def test_claude_model_version(model_id, expected):
    assert claude_model_version(model_id) == expected


# "Continue response": a conversation ending with an assistant message.

ENDS_WITH_ASSISTANT = [
    {"role": "user", "content": "Write a poem."},
    {"role": "assistant", "content": "Roses are red,"},
]


@pytest.mark.parametrize("model", [SONNET_3_7, SONNET_4_5])
def test_old_model_keeps_assistant_message_last(model):
    args = parse(model, ENDS_WITH_ASSISTANT)

    assert args["messages"][-1] == {"role": "assistant", "content": [{"text": "Roses are red,"}]}


@pytest.mark.parametrize("model", [OPUS_4_6, *NEW_MODELS])
def test_newer_model_gets_continue_prompt(model):
    args = parse(model, ENDS_WITH_ASSISTANT)

    assert args["messages"][-2] == {"role": "assistant", "content": [{"text": "Roses are red,"}]}
    assert args["messages"][-1] == CONTINUE_PROMPT


# temperature / top_p


def test_old_model_without_conflict_keeps_temperature_and_top_p():
    args = parse(SONNET_3_7, temperature=0.5, top_p=0.9)

    assert args["inferenceConfig"] == {"temperature": 0.5, "topP": 0.9}


@pytest.mark.parametrize("model", [SONNET_4_5, OPUS_4_6])
def test_conflict_model_keeps_temperature_and_drops_top_p(model):
    assert parse(model, temperature=0.5, top_p=0.9)["inferenceConfig"] == {"temperature": 0.5}
    assert parse(model, top_p=0.9)["inferenceConfig"] == {"topP": 0.9}


@pytest.mark.parametrize("model", NEW_MODELS)
def test_newer_model_drops_temperature_and_top_p(model):
    args = parse(model, temperature=0.5, top_p=0.9, max_tokens=1000)

    assert args["inferenceConfig"] == {"maxTokens": 1000}


# reasoning_effort


@pytest.mark.parametrize("model", [SONNET_4_5, OPUS_4_6])
def test_old_model_reasoning_effort_sends_budget_tokens(model):
    args = parse(model, reasoning_effort="medium", max_tokens=4096)

    assert args["additionalModelRequestFields"] == {
        "reasoning_config": {"type": "enabled", "budget_tokens": 2457}
    }


def test_old_model_reasoning_effort_without_max_tokens_is_rejected():
    with pytest.raises(HTTPException) as error:
        parse(SONNET_4_5, reasoning_effort="low")

    assert error.value.status_code == 400


@pytest.mark.parametrize("model", NEW_MODELS)
@pytest.mark.parametrize("max_tokens", [None, 4096])
def test_newer_model_ignores_reasoning_effort(model, max_tokens, caplog):
    caplog.set_level(logging.DEBUG, logger=bedrock_module.logger.name)

    args = parse(model, reasoning_effort="high", max_tokens=max_tokens)

    assert "additionalModelRequestFields" not in args
    assert [r.levelno for r in caplog.records if "reasoning_effort ignored" in r.getMessage()] == [logging.DEBUG]


def test_newer_model_still_passes_extra_body():
    args = parse(SONNET_5_5, reasoning_effort="high", extra_body={"thinking": {"type": "adaptive"}})

    assert args["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"}}
