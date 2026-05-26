import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# ── LLM singleton ─────────────────────────────────────────────────────────────
_llm = None

def get_llm() -> ChatMistralAI:
    global _llm
    if _llm is None:
        _llm = ChatMistralAI(
            model="mistral-small-latest",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.2,
        )
    return _llm


# ── Prompts (module-level constants) ─────────────────────────────────────────
ACTION_ITEMS_PROMPT = """You are an expert meeting analyst. From the meeting transcript, \
extract all action items. For each provide:
- Task description
- Owner (who is responsible)
- Deadline (if mentioned, else write 'Not specified')

Format as a numbered list. If none found say 'No action items found.'"""

KEY_DECISIONS_PROMPT = """You are an expert meeting analyst. From the meeting transcript, \
extract all key decisions made. Format as a numbered list. \
If none found say 'No key decisions found.'"""

OPEN_QUESTIONS_PROMPT = """From the meeting transcript, extract all unresolved questions \
or topics needing follow-up. Format as a numbered list. \
If none found say 'No open questions found.'"""

# Single combined prompt — replaces 3 separate LLM calls with 1
COMBINED_PROMPT = """You are an expert meeting analyst. Analyse the transcript below and return ALL THREE sections.

## ACTION ITEMS
List every action item. For each:
- Task description
- Owner (who is responsible)
- Deadline (if mentioned, else 'Not specified')
Numbered list. If none: 'No action items found.'

## KEY DECISIONS
List every key decision made.
Numbered list. If none: 'No key decisions found.'

## OPEN QUESTIONS
List every unresolved question or topic needing follow-up.
Numbered list. If none: 'No open questions found.'

Return exactly these three sections with the ## headers. Nothing else."""


# ── Chain builder ─────────────────────────────────────────────────────────────
def _build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return (
        RunnableLambda(lambda x: {"text": x})
        | prompt
        | get_llm()
        | StrOutputParser()
    )

# Cached chains — built once, reused across calls
_chains: dict = {}

def _get_chain(key: str, prompt: str):
    if key not in _chains:
        _chains[key] = _build_chain(prompt)
    return _chains[key]


# ── Individual extractors (kept for backwards compatibility) ──────────────────
def extract_action_items(transcript: str) -> str:
    return _get_chain("action_items", ACTION_ITEMS_PROMPT).invoke(transcript)

def extract_key_decisions(transcript: str) -> str:
    return _get_chain("key_decisions", KEY_DECISIONS_PROMPT).invoke(transcript)

def extract_questions(transcript: str) -> str:
    return _get_chain("open_questions", OPEN_QUESTIONS_PROMPT).invoke(transcript)


# ── Batch extractor — USE THIS in the pipeline (3 calls → 1) ─────────────────
def extract_all(transcript: str) -> dict[str, str]:
    """
    Single LLM call that returns all three extractions at once.
    Returns {"action_items": ..., "key_decisions": ..., "open_questions": ...}
    """
    chain = _get_chain("combined", COMBINED_PROMPT)
    raw   = chain.invoke(transcript)

    sections = {"action_items": "", "key_decisions": "", "open_questions": ""}
    current  = None

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "## ACTION ITEMS":
            current = "action_items"
        elif stripped == "## KEY DECISIONS":
            current = "key_decisions"
        elif stripped == "## OPEN QUESTIONS":
            current = "open_questions"
        elif current:
            sections[current] += line + "\n"

    return {k: v.strip() for k, v in sections.items()}
# Single combined prompt — 3 extractions in 1 LLM call
COMBINED_PROMPT = """You are an expert meeting analyst. Analyse the transcript below and return ALL THREE sections.

## ACTION ITEMS
List every action item. For each:
- Task description
- Owner (who is responsible)
- Deadline (if mentioned, else 'Not specified')
Numbered list. If none: 'No action items found.'

## KEY DECISIONS
List every key decision made.
Numbered list. If none: 'No key decisions found.'

## OPEN QUESTIONS
List every unresolved question or topic needing follow-up.
Numbered list. If none: 'No open questions found.'

Return exactly these three sections with the ## headers. Nothing else."""


def extract_all(transcript: str) -> dict[str, str]:
    chain = _get_chain("combined", COMBINED_PROMPT)
    raw   = chain.invoke(transcript)

    sections = {"action_items": "", "key_decisions": "", "open_questions": ""}
    current  = None

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "## ACTION ITEMS":
            current = "action_items"
        elif stripped == "## KEY DECISIONS":
            current = "key_decisions"
        elif stripped == "## OPEN QUESTIONS":
            current = "open_questions"
        elif current:
            sections[current] += line + "\n"

    return {k: v.strip() for k, v in sections.items()}