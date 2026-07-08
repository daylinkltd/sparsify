"""Web UI page invariants."""
import re

from sparsify.runtime.webui import PAGE


def test_no_emoji_icons():
    # icons must be SVG, not emoji/dingbats
    assert not re.search(r"[\U0001F300-\U0001FAFF☀-➿]", PAGE)


def test_structural_markers():
    for marker in ("svg class=\"ic\"", "id=\"toggleside\"", "id=\"backdrop\"",
                   "side-closed", "codeblock", "localStorage"):
        assert marker in PAGE, marker


def test_webui_has_settings_panel():
    from sparsify.runtime.webui import PAGE
    for marker in ('id="settings"', 'id="set-system"', 'id="set-temp"',
                   'id="set-theme"', 'withSystemPrompt', 'temperature'):
        assert marker in PAGE, marker
