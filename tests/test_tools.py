"""Built-in tool parsing/execution and web UI tool wiring."""
from sparsify.runtime import tools


def test_parse_single_and_multiple():
    v, calls = tools.parse_tool_calls(
        'ok<tool_call>{"name":"current_time","arguments":{}}</tool_call>')
    assert v == "ok" and calls == [{"name": "current_time", "arguments": {}}]
    _, calls = tools.parse_tool_calls(
        '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
        '<tool_call>{"name":"b","arguments":{}}</tool_call>')
    assert [c["name"] for c in calls] == ["a", "b"]


def test_parse_ignores_malformed():
    v, calls = tools.parse_tool_calls("text <tool_call>{bad json}</tool_call> end")
    assert calls == [] and "<tool_call>" not in v


def test_execute_unknown_and_time():
    assert "unknown tool" in tools.execute("nope", {})
    assert "UTC" in tools.execute("current_time", {})


def test_fetch_rejects_non_http():
    assert tools.execute("fetch_url", {"url": "file:///etc/passwd"}).startswith("error")


def test_github_dotgit_maps_to_raw_readme():
    cands = list(tools._github_raw_candidates("https://github.com/daylinkltd/sparsify.git"))
    assert any("raw.githubusercontent.com/daylinkltd/sparsify" in c
               and c.endswith("README.md") for c in cands)


def test_webui_has_tools_toggle():
    from sparsify.runtime.webui import PAGE
    assert 'id="toolsbtn"' in PAGE and 'tools: "auto"' in PAGE and "toolcard" in PAGE
