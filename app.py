import concurrent.futures
import os
from PIL import Image
import streamlit as st
from dotenv import load_dotenv

logo = Image.open("assets/logo.png")
# ── Page config must be first ─────────────────────────────────────────────────
st.set_page_config(
    page_title="ContextIQ",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()

# ── Lazy imports so Streamlit config runs first ───────────────────────────────
from utils.audio_processor import process_input, cleanup_chunks
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question_stream

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');
:root {
    --bg:         #080c14; --surface:  #0e1420; --surface2: #141b28;
    --border:     #1e2a3a; --accent:   #00e5ff; --accent2:  #7b61ff;
    --accent3:    #ff6b6b; --text:     #e8edf5; --text-muted:#5a6a82;
    --success:    #00c896; --warning:  #ffb547;
    --glow:       0 0 40px rgba(0,229,255,.12);
}
html,body,[class*="css"]{font-family:'DM Mono',monospace;background-color:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2.5rem 4rem!important;max-width:1280px;}
body::before{content:'';position:fixed;inset:0;z-index:-1;
  background:radial-gradient(ellipse 80% 60% at 10% -10%,rgba(0,229,255,.07) 0%,transparent 60%),
             radial-gradient(ellipse 60% 50% at 90% 10%,rgba(123,97,255,.06) 0%,transparent 55%),
             radial-gradient(ellipse 50% 40% at 50% 100%,rgba(255,107,107,.04) 0%,transparent 60%),var(--bg);
  animation:meshShift 14s ease-in-out infinite alternate;}
@keyframes meshShift{from{opacity:1}to{opacity:.8;filter:hue-rotate(15deg)}}
.hero{text-align:center;padding:3.5rem 0 2rem;}
.hero-badge{display:inline-block;font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;
  color:var(--accent);border:1px solid rgba(0,229,255,.3);border-radius:100px;
  padding:.3rem 1rem;margin-bottom:1.4rem;animation:fadeSlideDown .6s ease both;}
.hero h1{font-family:'Syne',sans-serif!important;font-size:clamp(2.4rem,5vw,4rem)!important;
  font-weight:800!important;letter-spacing:-.02em;line-height:1.1;
  background:linear-gradient(135deg,#e8edf5 30%,var(--accent) 70%,var(--accent2) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  margin:0 0 .8rem!important;animation:fadeSlideDown .7s ease .1s both;}
.hero p{color:var(--text-muted);font-size:.95rem;animation:fadeSlideDown .7s ease .2s both;}
@keyframes fadeSlideDown{from{opacity:0;transform:translateY(-14px)}to{opacity:1;transform:translateY(0)}}
.divider{height:1px;background:linear-gradient(90deg,transparent,var(--border) 20%,var(--accent2) 50%,var(--border) 80%,transparent);margin:2rem 0;opacity:.6;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
  padding:1.6rem 1.8rem;margin-bottom:1.2rem;position:relative;overflow:hidden;
  transition:border-color .25s,box-shadow .25s;animation:cardIn .5s ease both;}
.card:hover{border-color:rgba(0,229,255,.3);box-shadow:var(--glow);}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));opacity:0;transition:opacity .25s;}
.card:hover::before{opacity:1;}
@keyframes cardIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.card-label{font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);
  margin-bottom:.6rem;display:flex;align-items:center;gap:.5rem;}
.card-title{font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;color:var(--text);margin:0 0 .8rem;}
.card-body{color:#8fa0b8;font-size:.88rem;line-height:1.75;}
.tag{display:inline-block;font-size:.72rem;padding:.22rem .75rem;border-radius:100px;margin:.2rem .2rem 0 0;letter-spacing:.04em;}
.tag-cyan{background:rgba(0,229,255,.1);color:var(--accent);border:1px solid rgba(0,229,255,.25);}
.tag-purple{background:rgba(123,97,255,.1);color:var(--accent2);border:1px solid rgba(123,97,255,.25);}
.tag-red{background:rgba(255,107,107,.1);color:var(--accent3);border:1px solid rgba(255,107,107,.25);}
.tag-green{background:rgba(0,200,150,.1);color:var(--success);border:1px solid rgba(0,200,150,.25);}
.section-label{font-size:.65rem;letter-spacing:.22em;text-transform:uppercase;color:var(--text-muted);margin:2rem 0 .8rem;}
.stTextInput>div>div>input,.stTextArea>div>div>textarea{
  background:var(--surface2)!important;border:1px solid var(--border)!important;
  color:var(--text)!important;border-radius:10px!important;
  font-family:'DM Mono',monospace!important;font-size:.88rem!important;
  transition:border-color .2s,box-shadow .2s!important;}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{
  border-color:rgba(0,229,255,.5)!important;box-shadow:0 0 0 3px rgba(0,229,255,.08)!important;}
.stSelectbox>div>div{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important;}
.stButton>button{font-family:'Syne',sans-serif!important;font-weight:600!important;font-size:.88rem!important;
  letter-spacing:.05em;border-radius:10px!important;transition:all .22s ease!important;border:none!important;}
.stButton>button[kind="primary"],.stButton>button:first-child{
  background:linear-gradient(135deg,var(--accent),var(--accent2))!important;
  color:#080c14!important;padding:.6rem 1.8rem!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 24px rgba(0,229,255,.25)!important;filter:brightness(1.1)!important;}
.stButton>button:active{transform:translateY(0)!important;}
.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2))!important;border-radius:100px!important;}
.stProgress>div>div>div{background:var(--surface2)!important;border-radius:100px!important;}
.stSpinner>div{border-top-color:var(--accent)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:.3rem!important;gap:.2rem!important;}
.stTabs [data-baseweb="tab"]{font-family:'DM Mono',monospace!important;font-size:.78rem!important;letter-spacing:.06em!important;
  text-transform:uppercase!important;color:var(--text-muted)!important;border-radius:8px!important;padding:.5rem 1.1rem!important;transition:all .2s!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(0,229,255,.15),rgba(123,97,255,.15))!important;
  color:var(--accent)!important;border:1px solid rgba(0,229,255,.25)!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.4rem!important;}
.chat-bubble{padding:1rem 1.3rem;border-radius:14px;margin-bottom:.8rem;font-size:.875rem;line-height:1.7;animation:bubbleIn .3s ease both;}
.chat-bubble.user{background:linear-gradient(135deg,rgba(0,229,255,.1),rgba(123,97,255,.08));border:1px solid rgba(0,229,255,.2);margin-left:2rem;}
.chat-bubble.bot{background:var(--surface2);border:1px solid var(--border);margin-right:2rem;}
.chat-role{font-size:.62rem;letter-spacing:.2em;text-transform:uppercase;margin-bottom:.4rem;}
.chat-role.user{color:var(--accent);} .chat-role.bot{color:var(--accent2);}
@keyframes bubbleIn{from{opacity:0;transform:translateY(8px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}
.status-pill{display:inline-flex;align-items:center;gap:.45rem;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;padding:.3rem .9rem;border-radius:100px;}
.status-pill.running{background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.25);color:var(--accent);}
.status-pill.done{background:rgba(0,200,150,.08);border:1px solid rgba(0,200,150,.25);color:var(--success);}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor;animation:pulse 1.2s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
.step-row{display:flex;align-items:center;gap:.8rem;padding:.55rem 0;font-size:.82rem;color:var(--text-muted);transition:color .2s;}
.step-row.active{color:var(--accent);} .step-row.done{color:var(--success);}
.step-icon{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:.7rem;flex-shrink:0;border:1px solid currentColor;transition:all .3s;}
.step-row.done .step-icon{background:rgba(0,200,150,.15);border-color:var(--success);}
.step-row.active .step-icon{background:rgba(0,229,255,.15);border-color:var(--accent);animation:pulse 1s infinite;}
.transcript-box{background:var(--surface2);border:1px solid var(--border);border-left:3px solid var(--accent2);
  border-radius:10px;padding:1.2rem 1.4rem;font-size:.8rem;line-height:1.9;color:#7a8fa8;max-height:320px;overflow-y:auto;}
.transcript-box::-webkit-scrollbar{width:4px;} .transcript-box::-webkit-scrollbar-track{background:transparent;}
.transcript-box::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
.item-row{display:flex;gap:.9rem;align-items:flex-start;padding:.65rem 0;
  border-bottom:1px solid rgba(30,42,58,.6);font-size:.85rem;line-height:1.6;color:#8fa0b8;animation:cardIn .4s ease both;}
.item-row:last-child{border-bottom:none;}
.item-num{font-family:'DM Mono',monospace;font-size:.65rem;min-width:22px;height:22px;border-radius:6px;
  background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.2);color:var(--accent);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;}
[data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:1rem 1.2rem!important;}
[data-testid="stMetricLabel"]{color:var(--text-muted)!important;font-size:.72rem!important;letter-spacing:.1em!important;}
[data-testid="stMetricValue"]{font-family:'Syne',sans-serif!important;color:var(--accent)!important;font-size:1.6rem!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "result"       not in st.session_state: st.session_state.result       = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⬡ AI-Powered Intelligence</div>
    <h1>ContextIQ</h1>
    <p>Drop a YouTube URL or audio file — get instant intelligence from your meetings &amp; videos</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Input panel ───────────────────────────────────────────────────────────────
col_input, col_right = st.columns([1.6, 1], gap="large")

with col_input:
    st.markdown('<div class="section-label">⬡ Source</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "input_mode",
        options=["YouTube URL", "Upload File"],
        horizontal=True,
        label_visibility="collapsed",
    )

    source = None
    if input_mode == "YouTube URL":
        source = st.text_input(
            "source", placeholder="https://youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
    else:
        uploaded_file = st.file_uploader(
            "uploaded_file", type=["mp3", "mp4", "wav", "m4a", "webm"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            # Persist the upload to disk so the existing local-file
            # pipeline (convert_to_wav -> chunk_audio) can use it as-is.
            os.makedirs("downloads", exist_ok=True)
            saved_path = os.path.join("downloads", uploaded_file.name)
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source = saved_path

    st.markdown('<div class="section-label">⬡ Language</div>', unsafe_allow_html=True)
    language = st.selectbox("language", options=["english", "hinglish"], label_visibility="collapsed")
    run_btn = st.button("▶  Analyze", use_container_width=True)

with col_right:
    st.markdown("""
    <div class="card" style="animation-delay:.1s">
        <div class="card-label">⬡ What you get</div>
        <div style="display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.4rem">
            <span class="tag tag-cyan">Title</span><span class="tag tag-purple">Summary</span>
            <span class="tag tag-green">Action Items</span><span class="tag tag-red">Key Decisions</span>
            <span class="tag tag-cyan">Open Questions</span><span class="tag tag-purple">RAG Chat</span>
            <span class="tag tag-green">Full Transcript</span>
        </div>
        <div class="card-body" style="margin-top:1rem;font-size:.8rem">
            Supports YouTube links, MP3 / MP4 / WAV local files.<br>
            Groq Whisper for English &middot; Sarvam AI for Hinglish.<br>
            Semantic Q&amp;A powered by RAG + Mistral.
        </div>
    </div>""", unsafe_allow_html=True)

# ── Step definitions (matches new parallel pipeline) ─────────────────────────
STEPS = [
    ("🎙", "Extracting audio chunks"),
    ("✍️", "Transcribing  [Groq Whisper / Sarvam]"),
    ("🧠", "Parallel analysis  [title · summary · extraction]"),
    ("🔗", "Building RAG chain  [Chroma + Mistral]"),
]

def render_steps(current: int) -> str:
    html = '<div style="margin:.5rem 0">'
    for i, (icon, label) in enumerate(STEPS):
        cls  = "done" if i < current else ("active" if i == current else "")
        tick = "✓"   if i < current else ("→"      if i == current else "·")
        html += (f'<div class="step-row {cls}"><div class="step-icon">{tick}</div>'
                 f'<span>{icon} {label}</span></div>')
    return html + "</div>"

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_btn and source:
    st.session_state.result       = None
    st.session_state.chat_history = []
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.4], gap="large")
    with left_col:
        st.markdown('<div class="section-label">⬡ Pipeline</div>', unsafe_allow_html=True)
        status_box   = st.empty()
        steps_box    = st.empty()
        progress_bar = st.progress(0)
    with right_col:
        log_box = st.empty()

    def update(step_idx: int, log: str = ""):
        progress_bar.progress(int((step_idx / len(STEPS)) * 100))
        status_box.markdown(
            f'<div class="status-pill running"><div class="dot"></div>Step {step_idx}/{len(STEPS)}</div>',
            unsafe_allow_html=True)
        steps_box.markdown(render_steps(step_idx), unsafe_allow_html=True)
        if log:
            log_box.markdown(
                f'<div class="card" style="padding:1rem 1.2rem"><div class="card-label">⬡ Log</div>'
                f'<div class="card-body" style="font-size:.78rem">{log}</div></div>',
                unsafe_allow_html=True)

    try:
        update(0, "Loading source…")
        chunks = process_input(source)

        engine = "Sarvam AI" if language == "hinglish" else "Groq Whisper-large-v3"
        update(1, f"Got {len(chunks)} chunk(s). Transcribing via {engine}…")
        transcript = transcribe_all(chunks, language)
        cleanup_chunks(chunks)

        update(2, f"Transcript ready — {len(transcript):,} chars. Running parallel analysis…")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_title     = executor.submit(generate_title, transcript)
            future_summary   = executor.submit(summarize,      transcript)
            future_extracted = executor.submit(extract_all,    transcript)
            title     = future_title.result()
            summary   = future_summary.result()
            extracted = future_extracted.result()

        update(3, f"Analysis done. Title: <em>{title}</em><br>Building semantic index…")
        rag_chain = build_rag_chain(transcript)

        progress_bar.progress(100)
        status_box.markdown(
            '<div class="status-pill done"><div class="dot" style="animation:none"></div>Complete</div>',
            unsafe_allow_html=True)
        steps_box.markdown(render_steps(len(STEPS)), unsafe_allow_html=True)
        log_box.empty()

        st.session_state.result = {
            "title":          title,
            "transcript":     transcript,
            "summary":        summary,
            "action_items":   extracted["action_items"],
            "key_decisions":  extracted["key_decisions"],
            "open_questions": extracted["open_questions"],
            "rag_chain":      rag_chain,
        }

    except Exception as e:
        status_box.markdown(
            '<div class="status-pill" style="background:rgba(255,107,107,.1);border-color:rgba(255,107,107,.3);color:var(--accent3)">⚠ Error</div>',
            unsafe_allow_html=True)
        log_box.markdown(
            f'<div class="card"><div class="card-label">⬡ Error</div>'
            f'<div class="card-body" style="color:var(--accent3)">{e}</div></div>',
            unsafe_allow_html=True)

elif run_btn and not source:
    st.warning("Please enter a YouTube URL or upload a file first.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    words    = len(r["transcript"].split())
    ai_lines = len([l for l in r["action_items"].split("\n")   if l.strip()])
    kd_lines = len([l for l in r["key_decisions"].split("\n")  if l.strip()])
    oq_lines = len([l for l in r["open_questions"].split("\n") if l.strip()])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words", f"{words:,}")
    m2.metric("Action Items",   ai_lines)
    m3.metric("Key Decisions",  kd_lines)
    m4.metric("Open Questions", oq_lines)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="animation-delay:.05s">
        <div class="card-label">⬡ Title</div>
        <div class="card-title">{r['title']}</div>
    </div>""", unsafe_allow_html=True)

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs([
        "Summary", "Action Items", "Key Decisions", "Questions", "Transcript", "Chat"
    ])

    with tab_summary:
        st.markdown(f'<div class="card"><div class="card-label">⬡ Executive Summary</div>'
                    f'<div class="card-body">{r["summary"].replace(chr(10), "<br>")}</div></div>',
                    unsafe_allow_html=True)

    with tab_actions:
        items = [l.strip() for l in r["action_items"].split("\n") if l.strip()]
        rows  = "".join(f'<div class="item-row" style="animation-delay:{i*.05}s">'
                        f'<div class="item-num">{i+1}</div><div>{item}</div></div>'
                        for i, item in enumerate(items))
        st.markdown(f'<div class="card"><div class="card-label">⬡ Action Items</div>{rows}</div>', unsafe_allow_html=True)

    with tab_decisions:
        items = [l.strip() for l in r["key_decisions"].split("\n") if l.strip()]
        rows  = "".join(f'<div class="item-row" style="animation-delay:{i*.05}s">'
                        f'<div class="item-num" style="background:rgba(123,97,255,.08);border-color:rgba(123,97,255,.25);color:var(--accent2)">'
                        f'{i+1}</div><div>{item}</div></div>' for i, item in enumerate(items))
        st.markdown(f'<div class="card"><div class="card-label">⬡ Key Decisions</div>{rows}</div>', unsafe_allow_html=True)

    with tab_questions:
        items = [l.strip() for l in r["open_questions"].split("\n") if l.strip()]
        rows  = "".join(f'<div class="item-row" style="animation-delay:{i*.05}s">'
                        f'<div class="item-num" style="background:rgba(255,107,107,.08);border-color:rgba(255,107,107,.25);color:var(--accent3)">'
                        f'{i+1}</div><div>{item}</div></div>' for i, item in enumerate(items))
        st.markdown(f'<div class="card"><div class="card-label">⬡ Open Questions</div>{rows}</div>', unsafe_allow_html=True)

    with tab_transcript:
        st.markdown(f'<div class="section-label">⬡ Full Transcript</div>'
                    f'<div class="transcript-box">{r["transcript"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True)
        st.download_button("⬇  Download Transcript", data=r["transcript"],
                           file_name="transcript.txt", mime="text/plain")

    with tab_chat:
        st.markdown('<div class="section-label">⬡ Chat with your meeting</div>', unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            cls   = "user" if msg["role"] == "user" else "bot"
            label = "You"  if msg["role"] == "user" else "ContextIQ"
            st.markdown(f'<div class="chat-bubble {cls}"><div class="chat-role {cls}">{label}</div>'
                        f'{msg["content"]}</div>', unsafe_allow_html=True)

        q_col, btn_col = st.columns([5, 1], gap="small")
        with q_col:
            question = st.text_input("question", placeholder="Ask anything about the meeting…",
                                     label_visibility="collapsed", key="chat_input")
        with btn_col:
            send = st.button("Send", use_container_width=True)

        if send and question.strip():
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.markdown(f'<div class="chat-bubble user"><div class="chat-role user">You</div>'
                        f'{question}</div>', unsafe_allow_html=True)

            stream_placeholder = st.empty()
            full_answer = ""
            for token in ask_question_stream(r["rag_chain"], question):
                full_answer += token
                stream_placeholder.markdown(
                    f'<div class="chat-bubble bot"><div class="chat-role bot">ContextIQ</div>'
                    f'{full_answer}<span style="opacity:.4">▌</span></div>',
                    unsafe_allow_html=True)

            stream_placeholder.markdown(
                f'<div class="chat-bubble bot"><div class="chat-role bot">ContextIQ</div>'
                f'{full_answer}</div>', unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": full_answer})

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.result and not run_btn:
    st.markdown("""
    <div style="text-align:center;padding:4rem 0 2rem;color:var(--text-muted)">
        <div style="font-size:2.5rem;margin-bottom:1rem">⬡</div>
        <div style="font-family:'DM Mono',monospace;font-size:.8rem;letter-spacing:.1em">
            Enter a source above and click Analyze to begin
        </div>
    </div>""", unsafe_allow_html=True)