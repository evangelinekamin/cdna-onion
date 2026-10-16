"""Strict parsing of the grind protocol."""

import json

import pytest

from onionrig.rig import protocol as proto


def test_start_command_round_trips():
    line = proto.start_command(b"\x11" * 32, ("ab", "q"), 60.0)
    assert line.endswith("\n")
    message = json.loads(line)
    assert message == {"cmd": "start", "a0": "11" * 32, "targets": ["ab", "q"], "seconds": 60.0}
    assert json.loads(proto.start_command(b"\x00" * 32, ("a",), None))["seconds"] is None


@pytest.mark.parametrize("a0,targets,seconds", [
    (b"\x00" * 31, ("a",), 1.0),
    (b"\x00" * 32, (), 1.0),
    (b"\x00" * 32, tuple("a" * 257), 1.0),
    (b"\x00" * 32, ("a",), 0.0),
])
def test_start_command_rejects_bad_arguments(a0, targets, seconds):
    with pytest.raises(ValueError):
        proto.start_command(a0, targets, seconds)


def test_parse_events():
    assert proto.parse_event('{"event": "ready"}') == proto.Ready()
    assert proto.parse_event('{"event": "hits", "k": ["5", "-7", "0"]}') == proto.Hits((5, -7, 0))
    assert proto.parse_event('{"event": "done", "candidates": "123"}') == proto.Done(123)
    assert proto.parse_event('{"event": "error", "message": "boom"}') == proto.EntryError("boom")


@pytest.mark.parametrize("line", [
    "not json",
    "[1, 2]",
    '{"event": "dance"}',
    '{"event": "hits"}',
    '{"event": "hits", "k": [5]}',              # numbers lose precision; strings required
    '{"event": "hits", "k": ["1.5"]}',
    '{"event": "hits", "k": ["9223372036854775808"]}',  # 2^63
    '{"event": "hits", "k": ["--5"]}',
    '{"event": "hits", "k": ["-"]}',
    '{"event": "hits", "k": ["\\u0661\\u0662"]}',       # Arabic-Indic digits
    '{"event": "hits", "k": ["\\u00b2"]}',               # superscript two
    '{"event": "hits", "k": ["' + "9" * 5000 + '"]}',
    '{"event": "done"}',
    '{"event": "done", "candidates": "-1"}',
    '{"event": "done", "candidates": "\\u0661"}',
])
def test_parse_event_rejects_malformed(line):
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(line)


def test_offsets_at_int64_edges():
    edge = str(2**63 - 1)
    assert proto.parse_event(json.dumps({"event": "hits", "k": [edge, "-" + edge]})) \
        == proto.Hits((2**63 - 1, -(2**63 - 1)))


def test_parse_info():
    info = proto.parse_info(json.dumps({"contract": proto.CONTRACT, "name": "x",
                                        "version": "1", "model": "m"}) + "\n")
    assert (info.name, info.version, info.model, info.device) == ("x", "1", "m", "")


@pytest.mark.parametrize("text", [
    "",
    '{"contract": "onionrig-grind/1", "name": "x", "version": "1"}\nextra\n',
    '{"contract": "other/9", "name": "x", "version": "1"}',
    '{"contract": "onionrig-grind/1", "version": "1"}',
    '["not", "an", "object"]',
])
def test_parse_info_rejects(text):
    with pytest.raises(proto.ProtocolError):
        proto.parse_info(text)
