# ✍️ Upwork Proposal Writer

An AI-powered Streamlit app that generates personalized, human-sounding Upwork proposals using **RAG (Retrieval-Augmented Generation)** and **Google Gemini**.

Built for freelancer **Imran Riaz Chohan** — paste a job description, configure your settings, and get a copy-paste ready proposal in seconds.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]https://upworkproposal.streamlit.app/

---

## ✨ Features

- **RAG-powered context retrieval** — pulls the most relevant sections from your profile automatically
- **Google Gemini 2.5 Flash** — fast, high-quality proposal generation
- **Fully configurable** — tone, length, opening style, and portfolio selection
- **Human-sounding output** — prompt engineered to avoid generic, robotic language
- **Copy-paste ready** — clean text area for instant copying
- **Zero external vector DB** — in-memory numpy embeddings, no ChromaDB needed
- **Secrets-safe** — API key stored in Streamlit secrets, never in code

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector Search | NumPy cosine similarity (in-memory) |
| Language | Python 3.11+ |

---

## 📁 Project Structure

```
upwork-proposal-writer/
│
├── app.py                   # Main Streamlit application
├── profile.txt              # Freelancer profile (RAG knowledge base)
├── proposals.txt            # Past proposals for style reference (optional)
├── requirements.txt         # Python dependencies
│
└── .streamlit/
    ├── secrets.toml         # API keys (local only, never commit)
    └── config.toml          # Streamlit config (optional)
```

---

## ⚙️ How It Works

```
Job Description
      ↓
  Embed query (all-MiniLM-L6-v2)
      ↓
  Cosine similarity search over profile.txt chunks
      ↓
  Top-K relevant sections retrieved
      ↓
  Gemini prompt = retrieved context + job details + settings
      ↓
  Human-sounding proposal generated
      ↓
  Copy & paste → Upwork ✅
```

---

## 🔧 Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/upwork-proposal-writer.git
cd upwork-proposal-writer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

> Get your free API key at [aistudio.google.com](https://aistudio.google.com)

### 4. Run the app

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push your code to GitHub (**make sure `secrets.toml` is in `.gitignore`**)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Go to **Settings → Secrets** and add:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

5. Deploy ✅

---

## 📄 profile.txt Format

The app reads `profile.txt` as a structured knowledge base using `[SECTION]` tags:

```
[IDENTITY]
Name: Imran Riaz Chohan
Title: AI Chatbot & Full Stack MERN Developer

[SKILLS_AI]
- RAG chatbots trained on PDFs and knowledge bases
- AI assistants powered by OpenAI and LLMs

[PORTFOLIO_1]
Project Title: AI Customer Support Chatbot
Link: repumediaintelligence.com
...
```

Add or update any section — the RAG system will automatically pick up changes on next run.

---

## 📝 proposals.txt Format (Optional)

Add past winning proposals to help Gemini match your writing style:

```
Proposal-1:
Hi, This is exactly the kind of project...

Proposal-2:
Hi, Your requirement for a senior AI engineer...
```

---

## 🔒 Security

- API key is **never hardcoded** in source code
- `secrets.toml` is excluded via `.gitignore`
- No user data is stored or logged

---

## 🤝 About the Freelancer

**Imran Riaz Chohan**
AI Chatbot & Full Stack MERN Developer

- 💰 $25/hr
- ⭐ 5.0 rating (3 reviews)
- 🏆 100% Job Success Score
- 🔗 [repumediaintelligence.com](https://repumediaintelligence.com)
- 🔗 [globalpremiumtrades-inc.com](https://globalpremiumtrades-inc.com)
- 🔗 [alnoormdf.com](https://alnoormdf.com)

---

## 📜 License

MIT License — feel free to fork and adapt for your own freelancer profile.
