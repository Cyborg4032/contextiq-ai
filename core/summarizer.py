import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── LLM singleton ─────────────────────────────────────────────────────────────
_llm = None

def get_llm() -> ChatMistralAI:
    global _llm
    if _llm is None:
        _llm = ChatMistralAI(
            model="mistral-small-latest",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.3,
        )
    return _llm


# ── Prompts (module-level, built once) ────────────────────────────────────────
_MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize this portion of a meeting transcript concisely."),
    ("human", "{text}"),
])

_REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert meeting summarizer. Combine these partial summaries "
        "into one final professional meeting summary in bullet points.",
    ),
    ("human", "{text}"),
])

_TITLE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Based on the meeting transcript, generate a short professional meeting title "
        "(max 8 words). Only return the title, nothing else.",
    ),
    ("human", "{text}"),
])


# ── Chains (reuse singleton LLM) ──────────────────────────────────────────────
def _map_chain():
    return _MAP_PROMPT | get_llm() | StrOutputParser()

def _reduce_chain():
    return (
        RunnableLambda(lambda x: {"text": x})
        | _REDUCE_PROMPT
        | get_llm()
        | StrOutputParser()
    )

def _title_chain():
    return (
        RunnableLambda(lambda x: {"text": x})
        | _TITLE_PROMPT
        | get_llm()
        | StrOutputParser()
    )


# ── Text splitter singleton ───────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)

def split_transcript(transcript: str) -> list[str]:
    return _splitter.split_text(transcript)


# ── Public API ────────────────────────────────────────────────────────────────
def summarize(transcript: str) -> str:
    chunks = split_transcript(transcript)

    # Single chunk — skip map step entirely
    if len(chunks) == 1:
        return _reduce_chain().invoke(chunks[0])

    chain = _map_chain()

    # Parallel map: summarize all chunks concurrently
    chunk_summaries = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=min(len(chunks), 6)) as executor:
        futures = {
            executor.submit(chain.invoke, {"text": chunk}): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            i = futures[future]
            chunk_summaries[i] = future.result()

    combined = "\n\n".join(chunk_summaries)
    return _reduce_chain().invoke(combined)


def generate_title(transcript: str) -> str:
    # Only needs first 2000 chars
    return _title_chain().invoke(transcript[:2000])

