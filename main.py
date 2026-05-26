# main.py
import concurrent.futures
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question, ask_question_stream

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    print("🚀 Starting ContextIQ")
    print("─" * 60)

    # ── Step 1: Audio extraction ──────────────────────────────────────────────
    print("🎙  Extracting audio chunks ...")
    chunks = process_input(source)
    print(f"    -> {len(chunks)} chunk(s) ready")

    # ── Step 2: Transcription (Groq Whisper / Sarvam) ─────────────────────────
    engine = "Sarvam AI" if language == "hinglish" else "Groq Whisper-large-v3"
    print(f"\n✍️  Transcribing via {engine} ...")
    transcript = transcribe_all(chunks, language)
    print(f"    -> {len(transcript):,} characters transcribed")
    print(f"    -> Preview: {transcript[:200]} ...")

    # ── Step 3: Title + Summary + Extraction in parallel ─────────────────────
    # All three are independent — no reason to run sequentially
    print("\n🧠 Running parallel analysis (title · summary · extraction) ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_title     = executor.submit(generate_title, transcript)
        future_summary   = executor.submit(summarize,      transcript)
        future_extracted = executor.submit(extract_all,    transcript)

        title     = future_title.result()
        summary   = future_summary.result()
        extracted = future_extracted.result()

    print(f"    -> Title: {title}")
    print("    -> Summary, action items, decisions, questions done ✓")

    # ── Step 4: Vector store + RAG chain ──────────────────────────────────────
    print("\n🔗 Building RAG chain ...")
    rag_chain = build_rag_chain(transcript)
    print("    -> RAG chain ready ✓")
    print("\n" + "─" * 60)

    return {
        "title":          title,
        "transcript":     transcript,
        "summary":        summary,
        "action_items":   extracted["action_items"],
        "key_decisions":  extracted["key_decisions"],
        "open_questions": extracted["open_questions"],
        "rag_chain":      rag_chain,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    source   = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

    result = run_pipeline(source, language)

    print(f"\n{'=' * 60}")
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # ── RAG chat ──────────────────────────────────────────────────────────────
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue

        # Stream tokens to terminal as they arrive
        print("\n🤖 Assistant: ", end="", flush=True)
        for token in ask_question_stream(rag_chain, question):
            print(token, end="", flush=True)
        print("\n")