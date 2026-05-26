import os
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Groq limit is 25MB. At 32kbps mono MP3, 5 minutes ≈ 1.2MB — very safe.
CHUNK_MINUTES = 5


def download_youtube_audio(url: str) -> str:
    """Download YouTube audio and return path to WAV file."""
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        filename = filename.replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = CHUNK_MINUTES) -> list:
    """
    Split WAV into chunks and export each as MP3 at 32kbps mono.

    Why MP3 instead of WAV:
      - WAV  @ 16kHz mono = ~1.9 MB/min  → 10 min chunk = ~115 MB  (over Groq 25MB limit)
      - MP3  @ 32kbps mono = ~0.24 MB/min → 5 min chunk  = ~1.2 MB  (well under limit)

    Whisper accuracy is unaffected at 32kbps for speech.
    """
    audio    = AudioSegment.from_wav(wav_path)
    audio    = audio.set_channels(1).set_frame_rate(16000)  # normalise
    chunk_ms = chunk_minutes * 60 * 1000
    chunks   = []

    total = (len(audio) + chunk_ms - 1) // chunk_ms
    print(f"  Splitting into {total} chunk(s) of {chunk_minutes} min each...")

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk      = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.mp3"          # ← MP3 not WAV

        chunk.export(
            chunk_path,
            format="mp3",
            parameters=["-ac", "1",        # mono
                        "-ar", "16000",    # 16kHz
                        "-b:a", "32k"],    # 32kbps — tiny file, fine for speech
        )

        size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        print(f"  Chunk {i+1}/{total}: {size_mb:.1f} MB")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks