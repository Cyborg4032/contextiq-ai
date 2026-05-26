import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

# ── Shared prompt ─────────────────────────────────────────────────────────────
MEETING_ASSISTANT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
    ),
    ("human", "{question}"),
])


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


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain(retriever):
    """Assemble the LCEL RAG pipeline from any retriever."""
    return (
        {
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | MEETING_ASSISTANT_PROMPT
        | get_llm()
        | StrOutputParser()
    )


# ── Public API ────────────────────────────────────────────────────────────────
def build_rag_chain(transcript: str):
    """Build a fresh vector store from a new transcript and return a RAG chain."""
    vector_store = build_vector_store(transcript)
    retriever    = get_retriever(vector_store, k=4)
    return _build_chain(retriever)


def load_rag_chain():
    """Load a previously persisted vector store and return a RAG chain."""
    vector_store = load_vector_store()
    retriever    = get_retriever(vector_store, k=4)
    return _build_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    """Blocking call — returns full answer string."""
    return rag_chain.invoke(question)


def ask_question_stream(rag_chain, question: str):
    """
    Generator that yields answer tokens as they arrive.
    Use in Streamlit chat tab for token-by-token rendering.

        for token in ask_question_stream(rag_chain, question):
            placeholder.markdown(accumulated + token)
    """
    for chunk in rag_chain.stream(question):
        yield chunk