# Router: text-to-speech via Piper TTS

import io
import wave
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter(prefix="/tts", tags=["TTS"])

# Piper voice model location
_VOICE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "tts"
_VOICE_MODEL = "en_US-lessac-medium.onnx"
_piper_voice = None


def _get_piper():
    """Lazy-load the Piper voice."""
    global _piper_voice
    if _piper_voice is not None:
        return _piper_voice

    model_path = _VOICE_DIR / _VOICE_MODEL
    if not model_path.exists():
        return None

    from piper import PiperVoice  # type: ignore

    _piper_voice = PiperVoice.load(str(model_path))
    return _piper_voice


class SpeakRequest(BaseModel):
    text: str


@router.post("/speak", summary="Generate speech from text")
def speak(req: SpeakRequest):
    """
    Convert text to speech using Piper TTS.
    Returns WAV audio bytes.
    """
    voice = _get_piper()
    if voice is None:
        return Response(
            content=b"",
            media_type="audio/wav",
            headers={"X-TTS-Error": "Voice model not found"},
        )

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        voice.synthesize(req.text, wav)

    return Response(content=buf.getvalue(), media_type="audio/wav")
