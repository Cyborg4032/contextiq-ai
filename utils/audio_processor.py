import os
import yt_dlp

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


import subprocess

def convert_to_wav(input_path):
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            output_path,
        ],
        check=True,
    )

    return output_path


def chunk_audio(wav_path, chunk_minutes=5):
    chunk_length = chunk_minutes * 60
    chunks = []

    import subprocess

    duration = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                wav_path,
            ]
        )
        .decode()
        .strip()
    )

    total_chunks = int(duration // chunk_length) + 1

    for i in range(total_chunks):
        start = i * chunk_length
        output = f"{wav_path}_chunk_{i}.mp3"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-ss",
                str(start),
                "-t",
                str(chunk_length),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "32k",
                output,
            ],
            check=True,
        )

        if os.path.exists(output):
            chunks.append(output)

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
