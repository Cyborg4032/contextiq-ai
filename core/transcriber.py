import os
import requests
from pydub import AudioSegment
from groq import Groq

# ── Constants ─────────────────────────────────────────────────────────────────
SARVAM_PIECE_SECONDS = 25
SARVAM_API_KEY       = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL       = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL         = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GROQ_WHISPER_MODEL   = "whisper-large-v3"
GROQ_MAX_BYTES       = 24 * 1024 * 1024     # 24 MB safety margin


# ── Groq client singleton ─────────────────────────────────────────────────────
_groq_client = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in environment / .env")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── Groq (English) ────────────────────────────────────────────────────────────
def _send_to_groq(path: str) -> str:
    """Send a single audio file to Groq Whisper and return transcript text."""
    client   = get_groq_client()
    ext      = os.path.splitext(path)[-1].lower().lstrip(".")
    mime_map = {"mp3": "audio/mpeg", "wav": "audio/wav",
                "m4a": "audio/mp4",  "mp4": "audio/mp4",
                "webm": "audio/webm"}
    mime = mime_map.get(ext, "audio/mpeg")

    with open(path, "rb") as f:
        response = get_groq_client().audio.transcriptions.create(
            model=GROQ_WHISPER_MODEL,
            file=(os.path.basename(path), f, mime),   # ← explicit mime type
            language="en",
            response_format="text",
        )
    return response.strip() if isinstance(response, str) else response.text.strip()


def transcribe_chunk_groq(chunk_path: str) -> str:
    """
    Send one chunk to Groq Whisper.
    Auto-splits into 5-min sub-pieces if file still exceeds 24 MB.
    """
    if os.path.getsize(chunk_path) <= GROQ_MAX_BYTES:
        return _send_to_groq(chunk_path)

    # File still too large — split further into 5-min MP3 pieces
    audio    = AudioSegment.from_file(chunk_path)
    piece_ms = 5 * 60 * 1000
    n_pieces = (len(audio) + piece_ms - 1) // piece_ms
    texts    = []

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece_path = f"{chunk_path}_gq_{i}.mp3"
        audio[start : start + piece_ms].export(
            piece_path, format="mp3",
            parameters=["-ac", "1", "-ar", "16000", "-b:a", "32k"]
        )
        try:
            print(f"  → Groq sub-piece {i + 1}/{n_pieces} ...")
            texts.append(_send_to_groq(piece_path))
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(texts)


# ── Sarvam (Hinglish → English) ───────────────────────────────────────────────
def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=25s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}
    with open(piece_path, "rb") as f:
        files    = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data     = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_URL, headers=headers,
            files=files, data=data, timeout=120,
        )
    if not response.ok:
        print(f"\n Sarvam returned {response.status_code}: {response.text}\n")
        response.raise_for_status()
    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """Splits chunk into <=25s WAV pieces for Sarvam's sync API limit."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio    = AudioSegment.from_file(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    n_pieces = (len(audio) + piece_ms - 1) // piece_ms
    texts    = []

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece_path = f"{chunk_path}_sv_{i}.wav"
        audio[start : start + piece_ms].export(piece_path, format="wav")
        try:
            print(f"  → Sarvam piece {i + 1}/{n_pieces} ...")
            texts.append(_send_to_sarvam(piece_path))
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(texts)


# ── Router ────────────────────────────────────────────────────────────────────
def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_groq(chunk_path)


# ── Main entry point ──────────────────────────────────────────────────────────
def transcribe_all(chunks: list, language: str = "english") -> str:
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Groq (Whisper-large-v3)"
    print(f"Using {engine} for transcription.")

    parts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)} ...")
        parts.append(transcribe_chunk(chunk, language=language))

    print("Transcription complete.")
    return " ".join(parts).strip()