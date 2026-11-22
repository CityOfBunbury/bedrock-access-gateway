"""Tests for how tool settings and tool history are translated into a Converse request.

Bedrock rejects toolUse/toolResult content blocks unless toolConfig is set, and Converse has no
equivalent of tool_choice "none". When tools are disabled the gateway therefore sends no toolConfig
and turns any tool history into plain text.
"""

from unittest import mock

import pytest

# api.models.bedrock creates boto3 clients, lists models and loads the tiktoken
# encoding when imported. Stub those so the import makes no network calls.
with mock.patch("boto3.client"), mock.patch("tiktoken.get_encoding"):
    from api.models import bedrock as bedrock_module

from api.models.bedrock import BedrockModel
from api.schema import ChatRequest

MODEL_ID = "anthropic.claude-test-v1:0"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

TOOL_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Bunbury?"},
    {
        "role": "assistant",
        "content": "Let me check.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Bunbury"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 24C"},
    {"role": "user", "content": "Thanks, please summarise."},
]

PLAIN_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, how can I help?"},
    {"role": "user", "content": "Tell me a joke."},
]


@pytest.fixture(autouse=True)
def no_prompt_caching(monkeypatch):
    """Keep cachePoint blocks out of the request so only the tool handling is exercised."""
    monkeypatch.setattr(bedrock_module, "ENABLE_PROMPT_CACHING", False)


def parse(messages, **kwargs) -> dict:
    return BedrockModel()._parse_request(ChatRequest(model=MODEL_ID, messages=messages, **kwargs))


def content_keys(args: dict) -> set[str]:
    """Every content block type used across the Converse messages."""
    return {key for message in args["messages"] for block in message["content"] for key in block}


def assert_roles_alternate(args: dict):
    roles = [message["role"] for message in args["messages"]]
    assert roles[0] == "user"
    assert all(a != b for a, b in zip(roles, roles[1:]))


TOOL_HISTORY_AS_TEXT = [
    {"role": "user", "content": [{"text": "What is the weather in Bunbury?"}]},
    {
        "role": "assistant",
        "content": [
            {"text": "Let me check."},
            {"text": '[Tool call: get_weather({"city": "Bunbury"})]'},
        ],
    },
    {
        "role": "user",
        "content": [
            {"text": "[Tool result for get_weather]: Sunny, 24C"},
            {"text": "Thanks, please summarise."},
        ],
    },
]


@pytest.mark.parametrize("tools", [[], None])
def test_no_tools_with_tool_history_is_sent_as_text(tools):
    args = parse(TOOL_HISTORY, tools=tools, tool_choice="none")

    assert "toolConfig" not in args
    assert args["messages"] == TOOL_HISTORY_AS_TEXT
    assert_roles_alternate(args)


def test_tool_choice_none_with_tools_and_tool_history_is_sent_as_text():
    args = parse(TOOL_HISTORY, tools=[WEATHER_TOOL], tool_choice="none")

    assert "toolConfig" not in args
    assert args["messages"] == TOOL_HISTORY_AS_TEXT
    assert_roles_alternate(args)


def test_tools_disabled_keeps_cut_off_tool_call_arguments_as_received():
    messages = [
        {"role": "user", "content": "Write the report."},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": '{"path": "a.txt", "con'}},
                {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "Tool call was cut off"}]},
        {"role": "tool", "tool_call_id": "call_unknown", "content": "No such call"},
    ]

    args = parse(messages, tools=[], tool_choice="none")

    assert "toolConfig" not in args
    assert args["messages"] == [
        {"role": "user", "content": [{"text": "Write the report."}]},
        {
            "role": "assistant",
            "content": [
                {"text": '[Tool call: write_file({"path": "a.txt", "con)]'},
                {"text": "[Tool call: get_weather({})]"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"text": "[Tool result for write_file]: Tool call was cut off"},
                {"text": "[Tool result for call_unknown]: No such call"},
            ],
        },
    ]


@pytest.mark.parametrize(
    "tool_choice, expected",
    [
        ("auto", {"auto": {}}),
        ("required", {"any": {}}),
        ({"type": "function", "function": {"name": "get_weather"}}, {"tool": {"name": "get_weather"}}),
    ],
)
def test_tool_use_is_unchanged_when_tools_are_enabled(tool_choice, expected):
    args = parse(TOOL_HISTORY, tools=[WEATHER_TOOL], tool_choice=tool_choice)

    assert args["toolConfig"] == {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "inputSchema": {"json": WEATHER_TOOL["function"]["parameters"]},
                }
            }
        ],
        "toolChoice": expected,
    }
    assert args["messages"] == [
        {"role": "user", "content": [{"text": "What is the weather in Bunbury?"}]},
        {
            "role": "assistant",
            "content": [
                {"text": "Let me check."},
                {"toolUse": {"toolUseId": "call_1", "name": "get_weather", "input": {"city": "Bunbury"}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "call_1", "content": [{"text": "Sunny, 24C"}]}},
                {"text": "Thanks, please summarise."},
            ],
        },
    ]


def cut_off_history(arguments: str) -> list[dict]:
    return [
        {"role": "user", "content": "Write the report."},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": arguments}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Tool call was cut off"},
        {"role": "user", "content": "Try again."},
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param("", id="empty"),
        pytest.param('{"path": "a.txt", "con', id="truncated"),
        pytest.param('["a.txt"]', id="not-an-object"),
    ],
)
def test_tools_enabled_with_unusable_tool_call_arguments_sends_empty_input(arguments):
    args = parse(cut_off_history(arguments), tools=[WEATHER_TOOL])

    assert "toolConfig" in args
    assert args["messages"] == [
        {"role": "user", "content": [{"text": "Write the report."}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "call_1", "name": "write_file", "input": {}}}],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "call_1", "content": [{"text": "Tool call was cut off"}]}},
                {"text": "Try again."},
            ],
        },
    ]
    assert_roles_alternate(args)


def test_tools_enabled_with_valid_tool_call_arguments_is_unchanged():
    args = parse(cut_off_history('{"path": "a.txt", "content": "Done"}'), tools=[WEATHER_TOOL])

    tool_use = args["messages"][1]["content"][0]["toolUse"]
    assert tool_use["input"] == {"path": "a.txt", "content": "Done"}


def test_default_tool_choice_keeps_tools_enabled():
    args = parse(TOOL_HISTORY, tools=[WEATHER_TOOL])

    assert args["toolConfig"]["toolChoice"] == {"auto": {}}
    assert content_keys(args) == {"text", "toolUse", "toolResult"}


@pytest.mark.parametrize("extra", [{}, {"tools": []}, {"tools": [], "tool_choice": "none"}])
def test_no_tools_and_no_tool_history_is_unchanged(extra):
    args = parse(PLAIN_HISTORY, **extra)

    assert "toolConfig" not in args
    assert args["system"] == [{"text": "You are a helpful assistant."}]
    assert args["messages"] == [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi, how can I help?"}]},
        {"role": "user", "content": [{"text": "Tell me a joke."}]},
    ]
