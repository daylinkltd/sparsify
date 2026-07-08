"""Browser-tier registration, policy gating, and availability guard."""
from sparsify.runtime import tools
from sparsify.runtime.tools import ToolPolicy


def test_browser_tier_gated_off_by_default():
    pol = ToolPolicy.read_only()
    names = {t["function"]["name"] for t in tools.tools_for_policy(pol)}
    assert not any(n.startswith("browser_") for n in names)
    # execution is blocked even if a schema leaked
    assert tools.execute("browser_open", {"url": "https://example.com"}, pol).startswith("error")


def test_browser_tier_present_when_allowed(monkeypatch, tmp_path):
    monkeypatch.setattr("sparsify.runtime.browser.available", lambda: True)
    pol = ToolPolicy.from_flags(agent=True, workspace=tmp_path, allow_browser=True)
    names = {t["function"]["name"] for t in tools.tools_for_policy(pol)}
    assert {"browser_open", "browser_read", "browser_click", "browser_type"} <= names


def test_from_flags_skips_browser_when_engine_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("sparsify.runtime.browser.available", lambda: False)
    pol = ToolPolicy.from_flags(agent=True, workspace=tmp_path, allow_browser=True)
    assert pol.allow_browser is False  # asked for it, but engine not installed


def test_browser_schemas_wellformed():
    from sparsify.runtime.tools import _REGISTRY
    for name in ("browser_open", "browser_click", "browser_type"):
        schema, tier, impl = _REGISTRY[name]
        assert tier == "browser"
        assert schema["function"]["name"] == name
