import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind – PDF Q&A",
    page_icon="📄",
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
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state init
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ─────────────────────────────────────────────
#  Helper functions
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
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


def answer_question(vector_store, question):
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

    if hf_token:
        try:
            import requests
            prompt = f"Answer the question based on the context below.\nContext: {context}\n\nQuestion: {question}\nAnswer:"
            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.2}}
            response = requests.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-large",
                headers=headers, json=payload, timeout=30
            )
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            else:
                return f"📄 **Relevant passages from your PDF:**\n\n{context}"
        except Exception:
            return f"📄 **Relevant passages from your PDF:**\n\n{context}"
    else:
        return (
            "📄 **Most relevant passages from your PDF:**\n\n"
            + context
            + "\n\n---\n*Add your `HUGGINGFACEHUB_API_TOKEN` to a `.env` file to get AI-generated answers.*"
        )


def save_feedback(question, answer, rating, comment):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "pdf": st.session_state.pdf_name,
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment,
    }
    st.session_state.feedback_data.append(entry)
    with open("feedback_log.json", "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">📄 DocuMind</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">RAG-powered PDF Question Answering · Built with LangChain & Streamlit</p>', unsafe_allow_html=True)

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
            st.success(f"✅ '{uploaded_file.name}' indexed successfully!")
        st.markdown(f'<div class="status-ready">📗 Active PDF: {st.session_state.pdf_name}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
- Ask *"What is the main topic?"*
- Ask *"Summarize this document"*
- Ask *"List the key points"*
- Rate each answer using the feedback widget
""")

    if st.session_state.feedback_data:
        st.markdown("---")
        st.markdown("### 📊 Feedback Stats")
        total = len(st.session_state.feedback_data)
        avg_rating = sum(f["rating"] for f in st.session_state.feedback_data) / total
        st.metric("Responses rated", total)
        st.metric("Avg. rating", f"{avg_rating:.1f} / 5")

with right_col:
    st.markdown("### Chat with your PDF")

    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)
            with st.expander("💬 Rate this answer", expanded=False):
                rating = st.slider("Rating (1 = poor, 5 = excellent)", 1, 5, 3, key=f"rating_{i}")
                comment = st.text_input("Optional comment", key=f"comment_{i}", placeholder="What was good or bad?")
                if st.button("Submit Feedback", key=f"submit_{i}"):
                    question_for_fb = st.session_state.chat_history[i - 1]["content"] if i > 0 else ""
                    save_feedback(question_for_fb, msg["content"], rating, comment)
                    st.success("✅ Feedback saved – thank you!")

    st.markdown("---")

    if st.session_state.vector_store is None:
        st.info("⬅️ Upload a PDF first to start asking questions.")
    else:
        question = st.text_input(
            "Ask a question about your PDF",
            placeholder="e.g. What is the main topic of this document?",
            key="question_input",
        )
        ask_clicked = st.button("Ask ➜")

        if ask_clicked and question.strip():
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking…"):
                answer = answer_question(st.session_state.vector_store, question)
            st.session_state.chat_history.append({"role": "bot", "content": answer})
            st.rerun()