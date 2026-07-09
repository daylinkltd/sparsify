"""OpenAI function-calling interop: <tool_call> blocks become structured
tool_calls on the wire, and OpenAI-format history is normalized for the
chat template. This is the seam external agent frameworks (OpenClaw,
LangChain, any OpenAI SDK client) drive Sparsify through."""
import json
import queue

from sparsify.runtime import tools
from sparsify.runtime.server import EngineHost


# ── helpers ──────────────────────────────────────────────────────────────

def test_safe_visible_len_masks_tag_and_partial_prefix():
    assert tools.safe_visible_len("hello") == 5
    assert tools.safe_visible_len("hi<tool_call>{") == 2
    # a trailing partial prefix of the tag is held back...
    assert tools.safe_visible_len("hi<tool_c") == 2
    # ...but ordinary text ending in '<' unrelated to the tag is not
    assert tools.safe_visible_len("a < b") == 5


def test_openai_tool_calls_wire_format():
    out = tools.openai_tool_calls([{"name": "get_weather",
                                    "arguments": {"city": "Pune"}}])
    (tc,) = out
    assert tc["id"].startswith("call_") and tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Pune"}


def test_normalize_openai_messages():
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "f", "arguments": '{"x": 1}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
    ]
    norm = tools.normalize_openai_messages(msgs)
    assert norm[0]["content"] == ""  # never render a literal None
    assert norm[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert norm[1]["content"] == "42"
    assert msgs[0]["content"] is None  # input not mutated


def test_normalize_keeps_unparseable_arguments_string():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "f", "arguments": "{bad"}}]}]
    norm = tools.normalize_openai_messages(msgs)
    assert norm[0]["tool_calls"][0]["function"]["arguments"] == "{bad"


# ── worker passthrough: masked stream + structured calls ────────────────

class _FakeEngine:
    """chat_stream stand-in emitting text with an embedded tool call,
    split across chunks to exercise incremental masking."""

    def __init__(self, chunks):
        self._chunks = chunks

    def chat_stream(self, messages, max_tokens=None, tools=None,
                    temperature=0.0):
        for i, text in enumerate(self._chunks, 1):
            yield text, {"n_tokens": i, "finish_reason": None}


def _run_job(engine, tools_arg):
    """Run one job through the real EngineHost worker loop, with the
    heavy engine swapped for a fake."""
    import threading

    from sparsify.runtime.server import _Job

    host = EngineHost.__new__(EngineHost)  # skip __init__'s thread spawn
    host.policy = None
    host._get_engine = lambda tag: (engine, "fake")
    host._jobs = queue.Queue()
    job = _Job("fake", [{"role": "user", "content": "hi"}], None,
               tools=tools_arg)
    host._jobs.put(job)
    threading.Thread(target=EngineHost._worker, args=(host,),
                     daemon=True).start()
    items = []
    while True:
        item = job.out.get(timeout=10)
        items.append(item)
        if item[0] in ("done", "error"):
            break
    return items


def test_worker_emits_structured_calls_and_masks_text():
    engine = _FakeEngine([
        "The weather: ", "<tool_", 'call>{"name":"get_weather",'
        '"arguments":{"city":"Pune"}}</tool_call>'])
    items = _run_job(engine, tools_arg=[{"type": "function", "function": {
        "name": "get_weather", "parameters": {}}}])
    kinds = [i[0] for i in items]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    text = "".join(i[1] for i in items if i[0] == "chunk")
    assert "<tool_call>" not in text and "<tool_" not in text
    (calls,) = [i[1] for i in items if i[0] == "calls"]
    assert calls == [{"name": "get_weather", "arguments": {"city": "Pune"}}]


def test_worker_plain_text_unchanged_without_tools():
    engine = _FakeEngine(["hello ", "world"])
    items = _run_job(engine, tools_arg=None)
    text = "".join(i[1] for i in items if i[0] == "chunk")
    assert text == "hello world"
    assert not [i for i in items if i[0] == "calls"]
