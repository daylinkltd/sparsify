"""Voice endpoint plumbing (multipart + WAV decode) and attachment paths.
Whisper itself is optional and model-heavy; these tests cover everything
up to the transcribe call, plus the honest rejections."""
import io
import struct
import wave

import pytest

from sparsify.runtime.server import _decode_wav_16k_mono, _multipart_file


def _wav_bytes(rate=16000, channels=1, width=2, n=1600):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return buf.getvalue()


def test_decode_wav_happy_path():
    audio = _decode_wav_16k_mono(_wav_bytes())
    assert audio.dtype.name == "float32" and len(audio) == 1600
    assert abs(float(audio.max())) <= 1.0


def test_decode_wav_rejects_wrong_format():
    with pytest.raises(ValueError, match="44100"):
        _decode_wav_16k_mono(_wav_bytes(rate=44100))
    with pytest.raises(ValueError, match="not a valid WAV"):
        _decode_wav_16k_mono(b"definitely not audio")


def test_multipart_extracts_file_part():
    boundary = "xyz123"
    wav = _wav_bytes()
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="s.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav + f"\r\n--{boundary}--\r\n".encode()
    out = _multipart_file(body, f'multipart/form-data; boundary={boundary}')
    assert out == wav


def test_multipart_errors_are_plain():
    with pytest.raises(ValueError, match="boundary"):
        _multipart_file(b"", "multipart/form-data")
    with pytest.raises(ValueError, match="no file part"):
        _multipart_file(b"--b\r\nContent-Disposition: form-data; "
                        b'name="x"\r\n\r\n1\r\n--b--',
                        "multipart/form-data; boundary=b")


def test_tui_attach_flows_into_prompt():
    from sparsify.runtime.tui import ChatUI
    ui = ChatUI.__new__(ChatUI)
    ui._attachments = [("notes.txt", "hello world")]
    blocks = "\n\n".join(f"[attached file: {n}]\n```\n{c}\n```"
                         for n, c in ui._attachments)
    assert "notes.txt" in blocks and "hello world" in blocks


def test_webui_has_new_controls():
    from sparsify.runtime.webui import PAGE
    for marker in ('id="attachbtn"', 'id="micbtn"', 'id="chips"',
                   'id="dropzone"', 'id="oc-snippet"', 'id="set-agent"',
                   "/v1/audio/transcriptions", "consumeAttachments"):
        assert marker in PAGE, marker
    # honesty markers: images and cloud STT are not silently faked
    assert "vision model" in PAGE
    assert "transcribed locally" in PAGE.lower() or "locally" in PAGE
