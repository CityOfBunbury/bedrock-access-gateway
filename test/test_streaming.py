"""Tests for reading the ConverseStream event stream.

boto3 reads the event stream with blocking calls. Each read must run in the thread pool, otherwise one
stream waiting for Bedrock stalls every other request handled by the same process.
"""

import asyncio
import time
from unittest import mock

# api.models.bedrock creates boto3 clients, lists models and loads the tiktoken
# encoding when imported. Stub those so the import makes no network calls.
with mock.patch("boto3.client"), mock.patch("tiktoken.get_encoding"):
    from api.models import bedrock as bedrock_module  # noqa: F401

from api.models.bedrock import BedrockModel

CHUNKS = [
    {"messageStart": {"role": "assistant"}},
    {"contentBlockDelta": {"delta": {"text": "Hello"}, "contentBlockIndex": 0}},
    {"messageStop": {"stopReason": "end_turn"}},
]


class SlowStream:
    """Stands in for the boto3 event stream: every read blocks before returning the next event."""

    def __init__(self, chunks, delay):
        self.chunks = chunks
        self.delay = delay

    def __iter__(self):
        for chunk in self.chunks:
            time.sleep(self.delay)
            yield chunk


def test_slow_stream_does_not_block_other_coroutines():
    async def main():
        ticks = 0
        finished = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not finished.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        received = [chunk async for chunk in BedrockModel()._async_iterate(SlowStream(CHUNKS, delay=0.2))]
        finished.set()
        await task
        return received, ticks

    received, ticks = asyncio.run(main())

    assert received == CHUNKS
    # The stream spends about 0.6s waiting. If those waits ran on the event loop the
    # ticker would only get a turn between reads, a handful of ticks at most.
    assert ticks >= 15


def test_empty_stream_yields_nothing():
    async def main():
        return [chunk async for chunk in BedrockModel()._async_iterate(SlowStream([], delay=0))]

    assert asyncio.run(main()) == []
