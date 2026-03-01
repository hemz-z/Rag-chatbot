import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
import os
import re
import json
from datetime import datetime

# ─────────────────────────────────────────────
#  Add your Gemini API key here
# ─────────────────────────────────────────────
GROQ_API_KEY = "----key----"
client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f0f13;
    color: #e8e8f0;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.subtitle {
    color: #9ca3af;
    font-size: 1rem;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}
.chat-bubble-user {
    background: linear-gradient(135deg, #6d28d9, #4f46e5);
    color: white;
    padding: 0.8rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
}
.chat-bubble-bot {
    background: #1e1e2e;
    border: 1px solid #2d2d42;
    color: #e8e8f0;
    padding: 0.8rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 0;
    max-width: 85%;
    font-size: 0.95rem;
    line-height: 1.6;
}
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}
.status-ready {
    background: #052e16;
    border: 1px solid #16a34a;
    color: #4ade80;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    display: inline-block;
    margin-bottom: 0.5rem;
}
.summary-box {
    background: #1e1e2e;
    border: 1px solid #2d2d42;
    border-left: 4px solid #a78bfa;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    line-height: 1.8;
    font-size: 0.95rem;
    margin-top: 1rem;
}
.yt-info-box {
    background: #1a1a2e;
    border: 1px solid #2d2d42;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state init
# ─────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "vector_store": None,
    "feedback_data": [],
    "pdf_name": None,
    "yt_summary": None,
    "yt_transcript": None,
    "yt_video_id": None,
    "yt_feedback_data": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Helper functions — Gemini
# ─────────────────────────────────────────────

def call_gemini(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Groq error:", e)
        return None

def save_feedback(source, question, answer, rating, comment, feedback_list_key):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment,
    }
    st.session_state[feedback_list_key].append(entry)
    with open("feedback_log.json", "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────
#  Helper functions — PDF
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)


def answer_question(vector_store, question):
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""
    result = call_gemini(prompt)
    if result:
        return result
    return "📄 **Relevant passages from your PDF:**\n\n" + context


# ─────────────────────────────────────────────
#  Helper functions — YouTube
# ─────────────────────────────────────────────

def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_transcript(video_id):
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.fetch(video_id)
        full_text = " ".join([entry.text for entry in transcript_list])
        return full_text
    except Exception:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry["text"] for entry in transcript_list])
        return full_text


def summarize_transcript(transcript):
    trimmed = transcript[:8000]
    prompt = f"""You are a helpful assistant. Please provide a clear, well-structured summary of the following video transcript.
Include:
- Main topic
- Key points
- Important takeaways

Transcript:
{trimmed}

Summary:"""
    result = call_gemini(prompt)
    if result:
        return result
    return "📝 **Transcript excerpt:**\n\n" + transcript[:1000] + "..."


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🧠 DocuMind</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">RAG-powered PDF Q&A & YouTube Summarizer · Built with LangChain, Gemini & Streamlit</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📄 PDF Question Answering", "▶️ YouTube Summarizer"])


# ══════════════════════════════════════════════
#  TAB 1 — PDF Q&A
# ══════════════════════════════════════════════
with tab1:
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown("### Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

        if uploaded_file:
            if st.session_state.pdf_name != uploaded_file.name:
                with st.spinner("🔍 Reading and indexing your PDF…"):
                    raw_text = extract_text_from_pdf(uploaded_file)
                    st.session_state.vector_store = build_vector_store(raw_text)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                st.success(f"✅ '{uploaded_file.name}' indexed!")
            st.markdown(f'<div class="status-ready">📗 Active: {st.session_state.pdf_name}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
- *"What is the main topic?"*
- *"Summarize this document"*
- *"List the key points"*
- *"What does chapter 2 say?"*
""")

        if st.session_state.feedback_data:
            st.markdown("---")
            st.markdown("### 📊 Feedback Stats")
            total = len(st.session_state.feedback_data)
            avg = sum(f["rating"] for f in st.session_state.feedback_data) / total
            st.metric("Responses rated", total)
            st.metric("Avg. rating", f"{avg:.1f} / 5")

    with right_col:
        st.markdown("### Chat with your PDF")

        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)
                with st.expander("💬 Rate this answer", expanded=False):
                    rating = st.slider("Rating", 1, 5, 3, key=f"pdf_rating_{i}")
                    comment = st.text_input("Comment (optional)", key=f"pdf_comment_{i}", placeholder="What was good or bad?")
                    if st.button("Submit Feedback", key=f"pdf_submit_{i}"):
                        q = st.session_state.chat_history[i - 1]["content"] if i > 0 else ""
                        save_feedback(st.session_state.pdf_name, q, msg["content"], rating, comment, "feedback_data")
                        st.success("✅ Feedback saved!")

        st.markdown("---")

        if st.session_state.vector_store is None:
            st.info("⬅️ Upload a PDF first to start asking questions.")
        else:
            question = st.text_input(
                "Ask a question",
                placeholder="e.g. What is the main topic of this document?",
                key="pdf_question_input",
            )
            if st.button("Ask ➜", key="pdf_ask"):
                if question.strip():
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    with st.spinner("Thinking…"):
                        answer = answer_question(st.session_state.vector_store, question)
                    st.session_state.chat_history.append({"role": "bot", "content": answer})
                    st.rerun()


# ══════════════════════════════════════════════
#  TAB 2 — YouTube Summarizer
# ══════════════════════════════════════════════
with tab2:
    left_col2, right_col2 = st.columns([1, 2], gap="large")

    with left_col2:
        st.markdown("### Paste YouTube URL")
        yt_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="yt_url_input"
        )

        summarize_clicked = st.button("Summarize ▶️", key="yt_summarize")

        if summarize_clicked and yt_url.strip():
            video_id = extract_video_id(yt_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL. Please check and try again.")
            else:
                with st.spinner("📥 Fetching transcript…"):
                    try:
                        transcript = get_transcript(video_id)
                        st.session_state.yt_transcript = transcript
                        st.session_state.yt_video_id = video_id
                    except Exception as e:
                        st.error(f"❌ Could not fetch transcript. Make sure the video has captions enabled.\n\nError: {str(e)}")
                        transcript = None

                if transcript:
                    with st.spinner("🧠 Summarizing…"):
                        summary = summarize_transcript(transcript)
                        st.session_state.yt_summary = summary
                    st.success("✅ Summary ready!")

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
- Works on videos with **captions/subtitles**
- Auto-generated captions work too
- Best for lectures, tutorials, talks
- Try TED Talks or educational videos
""")

        if st.session_state.yt_feedback_data:
            st.markdown("---")
            st.markdown("### 📊 Feedback Stats")
            total = len(st.session_state.yt_feedback_data)
            avg = sum(f["rating"] for f in st.session_state.yt_feedback_data) / total
            st.metric("Summaries rated", total)
            st.metric("Avg. rating", f"{avg:.1f} / 5")

    with right_col2:
        st.markdown("### Video Summary")

        if st.session_state.yt_video_id:
            st.markdown(f"""
            <div class="yt-info-box">
                <iframe width="100%" height="250"
                src="https://www.youtube.com/embed/{st.session_state.yt_video_id}"
                frameborder="0" allowfullscreen></iframe>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.yt_summary:
            st.markdown(f'<div class="summary-box">{st.session_state.yt_summary}</div>', unsafe_allow_html=True)

            with st.expander("📜 View full transcript"):
                st.text_area("Transcript", st.session_state.yt_transcript, height=300, label_visibility="collapsed")

            st.markdown("---")
            st.markdown("#### 💬 Rate this summary")
            yt_rating = st.slider("Rating (1 = poor, 5 = excellent)", 1, 5, 3, key="yt_rating")
            yt_comment = st.text_input("Comment (optional)", key="yt_comment", placeholder="Was the summary accurate?")
            if st.button("Submit Feedback", key="yt_feedback_submit"):
                save_feedback(
                    f"youtube:{st.session_state.yt_video_id}",
                    "summarize video",
                    st.session_state.yt_summary,
                    yt_rating, yt_comment,
                    "yt_feedback_data"
                )
                st.success("✅ Feedback saved – thank you!")
        else:
            st.info("⬅️ Paste a YouTube URL and click Summarize to get started.")