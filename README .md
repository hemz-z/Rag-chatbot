# 🧠 Clario — AI-Powered Document & Video Assistant

> RAG-based PDF Question Answering and YouTube Video Summarizer built with LangChain, Streamlit, and HuggingFace.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.1-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

- 📄 **PDF Question Answering** — Upload any PDF and ask questions about it in natural language
- ▶️ **YouTube Summarizer** — Paste a YouTube URL and get an instant AI-generated summary from its captions
- 🔍 **RAG Pipeline** — Uses Retrieval Augmented Generation for accurate, context-aware answers
- 💬 **Feedback Mechanism** — Rate every answer/summary and track average ratings
- 🎨 **Clean UI** — Dark-themed, modern interface built with Streamlit

---

## 🖼️ Screenshots

> PDF Q&A Tab

![PDF Tab](https://via.placeholder.com/800x400?text=PDF+Question+Answering)

> YouTube Summarizer Tab

![YouTube Tab](https://via.placeholder.com/800x400?text=YouTube+Summarizer)

---

## 🏗️ How It Works

```
📄 PDF Upload                        ▶️ YouTube URL
      ↓                                     ↓
Extract Text (PyPDF2)           Fetch Captions (youtube-transcript-api)
      ↓                                     ↓
Split into Chunks (LangChain)       Trim & Prepare Text
      ↓                                     ↓
Embed with HuggingFace          Send to HuggingFace LLM
(sentence-transformers)                     ↓
      ↓                              AI-Generated Summary
Store in FAISS Vector DB
      ↓
User Asks a Question
      ↓
Similarity Search → Retrieve Chunks
      ↓
Send to HuggingFace LLM (flan-t5)
      ↓
Display Answer + Collect Feedback
```

This is called **RAG — Retrieval Augmented Generation**.
The model doesn't memorize your PDF — it looks up the most relevant parts on demand.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or above
- VS Code (recommended)
- A free [HuggingFace](https://huggingface.co) account

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/clario.git
cd clario
```

**2. Create a virtual environment**
```bash
python -m venv venv
```

**3. Activate the virtual environment**

On Windows:
```bash
venv\Scripts\activate
```
On Mac/Linux:
```bash
source venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Set up your HuggingFace API token**

Create a `.env` file in the project root:
```
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```
Get your free token at 👉 https://huggingface.co/settings/tokens

**6. Run the app**
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📦 Tech Stack

| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI |
| [LangChain](https://langchain.com) | RAG pipeline & text splitting |
| [HuggingFace](https://huggingface.co) | Embeddings & LLM (flan-t5) |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search |
| [PyPDF2](https://pypdf2.readthedocs.io) | PDF text extraction |
| [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) | YouTube captions |
| [sentence-transformers](https://www.sbert.net) | Text embeddings |

---

## 📁 Project Structure

```
clario/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env                # Your API token (do NOT commit this)
├── .env.example        # Token template
├── feedback_log.json   # Auto-generated feedback log
└── README.md           # This file
```

---

## 🔮 Roadmap

- [x] PDF Question Answering
- [x] YouTube Video Summarizer
- [x] Feedback Mechanism
- [ ] Chat history export as PDF
- [ ] Support for multiple PDFs at once
- [ ] Better LLM (GPT-4 / Gemini integration)
- [ ] Feedback export as CSV
- [ ] User authentication

---

## ⚠️ Notes

- YouTube summarizer only works on videos that have **captions/subtitles** enabled
- The app works without a HuggingFace token — it will show relevant passages instead of generated answers
- Do **not** upload your `.env` file to GitHub — it contains your secret token

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Built With ❤️ using LangChain & Streamlit
