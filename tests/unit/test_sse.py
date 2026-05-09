"""StreamAccumulator: SSE delta parsing for OpenAI-compat streams."""

from __future__ import annotations

from slancha_local.proxy.sse import StreamAccumulator


def test_simple_stream():
    acc = StreamAccumulator()
    acc.feed(b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n')
    acc.feed(b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n')
    acc.feed(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
    acc.feed(b"data: [DONE]\n\n")
    assert acc.content == "Hi there"
    assert acc.delta_count == 2
    assert acc.finish_reason == "stop"


def test_chunks_split_mid_block():
    acc = StreamAccumulator()
    acc.feed(b'data: {"choices":[{"delta":{"content":"He')
    acc.feed(b'llo"}}]}\n\n')
    acc.feed(b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n')
    assert acc.content == "Hello world"
    assert acc.delta_count == 2


def test_malformed_json_is_skipped():
    acc = StreamAccumulator()
    acc.feed(b"data: not-json\n\n")
    acc.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
    assert acc.content == "hi"
    assert acc.delta_count == 1


def test_done_sentinel_stops_processing():
    acc = StreamAccumulator()
    acc.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
    acc.feed(b"data: [DONE]\n\n")
    acc.feed(b'data: {"choices":[{"delta":{"content":" extra"}}]}\n\n')
    assert acc.content == "hi"
    assert acc.delta_count == 1


def test_usage_block_is_captured():
    acc = StreamAccumulator()
    payload = (
        b'data: {"choices":[{"delta":{"content":"x"}}],'
        b' "usage":{"prompt_tokens":42, "completion_tokens":7}}\n\n'
    )
    acc.feed(payload)
    assert acc.usage_in == 42
    assert acc.usage_out == 7
    assert acc.tokens_out_estimate == 7


def test_tokens_out_estimate_falls_back_to_delta_count():
    acc = StreamAccumulator()
    for word in ["a", " b", " c", " d"]:
        acc.feed(b'data: {"choices":[{"delta":{"content":"' + word.encode() + b'"}}]}\n\n')
    assert acc.usage_out == 0
    assert acc.tokens_out_estimate == 4


def test_crlf_line_endings():
    acc = StreamAccumulator()
    acc.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}\r\n\r\n')
    assert acc.content == "hi"


def test_multiple_blocks_in_one_feed():
    acc = StreamAccumulator()
    body = (
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"c"}}]}\n\n'
    )
    acc.feed(body)
    assert acc.content == "abc"
    assert acc.delta_count == 3


def test_empty_feed_is_safe():
    acc = StreamAccumulator()
    acc.feed(b"")
    assert acc.content == ""
