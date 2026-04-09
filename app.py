# import streamlit as st
# from google import genai
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import re, os, warnings, logging

# warnings.filterwarnings("ignore")
# logging.getLogger("transformers").setLevel(logging.ERROR)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # ─────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────
# PROFILE_FILE   = "./profile.txt"
# PROPOSALS_FILE = "./proposals.txt"

# gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# st.set_page_config(page_title="Proposal Writer", page_icon="✍️", layout="wide")

# # ─────────────────────────────────────────────
# # EMBEDDING MODEL — ek baar load
# # ─────────────────────────────────────────────
# @st.cache_resource(show_spinner="📦 Embedding model load ho raha hai (sirf ek baar)...")
# def load_embed_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# # ─────────────────────────────────────────────
# # CHUNKS LOAD
# # ─────────────────────────────────────────────
# def load_profile_chunks(filepath: str) -> list[dict]:
#     if not os.path.exists(filepath):
#         return []
#     with open(filepath, "r", encoding="utf-8") as f:
#         raw = f.read()
#     parts = re.split(r"\[([A-Z_0-9]+)\]", raw)
#     chunks = []
#     for i in range(1, len(parts) - 1, 2):
#         sec     = parts[i].strip()
#         content = parts[i + 1].strip()
#         if content:
#             chunks.append({"section": sec, "content": f"[{sec}]\n{content}"})
#     return chunks

# def load_proposal_chunks(filepath: str) -> list[dict]:
#     if not os.path.exists(filepath):
#         return []
#     with open(filepath, "r", encoding="utf-8") as f:
#         raw = f.read()
#     parts = re.split(r"(Proposal-\d+:)", raw)
#     chunks = []
#     for i in range(1, len(parts) - 1, 2):
#         label   = parts[i].strip()
#         content = parts[i + 1].strip()
#         if content:
#             chunks.append({"section": label, "content": f"[EXAMPLE_PROPOSAL]\n{content}"})
#     return chunks

# # ─────────────────────────────────────────────
# # IN-MEMORY INDEX — numpy cosine similarity
# # ─────────────────────────────────────────────
# @st.cache_resource(show_spinner="🔧 Profile index ban raha hai...")
# def build_index(_model):
#     profile_chunks  = load_profile_chunks(PROFILE_FILE)
#     proposal_chunks = load_proposal_chunks(PROPOSALS_FILE)
#     all_chunks      = profile_chunks + proposal_chunks

#     docs = [c["content"] for c in all_chunks]
#     embeddings = _model.encode(docs, normalize_embeddings=True)  # shape: (N, 384)

#     return all_chunks, embeddings

# def retrieve(query: str, model, all_chunks, embeddings, top_k=5):
#     q_emb = model.encode([query], normalize_embeddings=True)[0]  # (384,)

#     # Cosine similarity = dot product (already normalized)
#     scores = np.dot(embeddings, q_emb)  # (N,)
#     top_indices = np.argsort(scores)[::-1][:top_k]

#     ctx, dbg = [], []
#     for idx in top_indices:
#         ctx.append(all_chunks[idx]["content"])
#         dbg.append({
#             "section":   all_chunks[idx]["section"],
#             "relevance": f"{round(float(scores[idx]) * 100, 1)}%"
#         })
#     return "\n\n---\n\n".join(ctx), dbg

# # ─────────────────────────────────────────────
# # PROPOSAL GENERATOR
# # ─────────────────────────────────────────────
# PORTFOLIO_LINKS = """
# - AI Customer Support & Lead Generation Chatbot → repumediaintelligence.com
# - AI Medical Symptom Checker Chatbot            → (mobile app, no public link)
# - Global Premium Trades Inc - E-commerce        → globalpremiumtrades-inc.com
# - Al-Noor MDF Board Industries - Website        → alnoormdf.com
# - RAG AI Knowledge Assistant for Agriculture    → (research project, no public link)
# """.strip()

# def generate_proposal(ctx: str, job: dict, settings: dict) -> str:
#     greeting = f"Hi {job['client_name']}," if job['client_name'] else "Hi,"

#     portfolio_line = (
#         "Mention ONLY these portfolio projects (with exact links where available): "
#         + settings["selected_portfolio"]
#     ) if settings["selected_portfolio"] else "Do not mention portfolio projects."

#     budget_line = (
#         f"Client budget is {job['budget']} — mention $25/hr rate naturally."
#         if job["budget"] else "Do not mention hourly rate."
#     )

#     prompt = f"""You are an expert Upwork proposal writer. Write a SHORT, HIGH-CONVERTING, human-sounding proposal for freelancer Imran Riaz Chohan.

# == FREELANCER PROFILE (retrieved) ==
# {ctx}

# == AVAILABLE PORTFOLIO LINKS ==
# {PORTFOLIO_LINKS}

# == JOB DETAILS ==
# Title      : {job['title']}
# Description: {job['description']}
# Budget     : {job['budget'] or 'Not specified'}

# == WRITING INSTRUCTIONS ==
# - Start with exactly: "{greeting}"
# - Opening style      : {settings['opening_style']}
# - Tone               : {settings['tone']}
# - Target length      : {settings['length']}
# - {portfolio_line}
# - {budget_line}
# - Match tech stack tightly to job requirements — only mention relevant skills
# - NO generic filler like "I am passionate" or "I am a hard worker"
# - Be specific, confident, direct — write like a senior freelancer
# - First 2 sentences must show you understand the client's exact problem
# - End with a clear, natural call to action
# - Sound human — vary sentence length, avoid robotic patterns
# - Final 2 lines must be exactly:
# Best Regards,
# Imran Riaz

# Write only the proposal. No commentary, no preamble."""

#     response = gemini_client.models.generate_content(
#         model="gemini-2.5-flash-lite",
#         contents=prompt
#     )
#     return response.text

# # ─────────────────────────────────────────────
# # INIT
# # ─────────────────────────────────────────────
# embed_model = load_embed_model()
# all_chunks, embeddings = build_index(embed_model)

# # ─────────────────────────────────────────────
# # SIDEBAR
# # ─────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 👤 Imran Riaz Chohan")
#     st.markdown("*AI Chatbot & Full Stack MERN Developer*")
#     st.markdown("💰 **$25/hr** | ⭐ 5.0 | 🏆 100% JSS")
#     st.markdown("---")
#     st.markdown("**🔗 Portfolio**")
#     st.markdown("[AI Support Chatbot](https://repumediaintelligence.com)")
#     st.markdown("[Global Premium Trades-inc](https://globalpremiumtrades-inc.com)") 
#     st.markdown("[Global Premium Trades](https://globalpremiumtrades.com)")
#     st.markdown("[Al-Noor MDF](https://alnoormdf.com)")

# # ─────────────────────────────────────────────
# # MAIN UI
# # ─────────────────────────────────────────────
# st.title("✍️ Upwork Proposal Writer")
# st.caption("Job details bharo → RAG + Gemini → copy-paste ready proposal")
# st.markdown("---")

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.subheader("📋 Job Details")
#     job_title   = st.text_input("Job Title *", placeholder="e.g. React Developer for SaaS Platform")
#     job_desc    = st.text_area("Job Description *", placeholder="Client ki full job posting yahan paste karo...", height=240)
#     client_name = st.text_input("Client Name (optional)", placeholder="e.g. John, Sarah")
#     budget      = st.text_input("Budget (optional)", placeholder="e.g. $500, $30/hr")

# with col2:
#     st.subheader("⚙️ Settings")
#     tone = st.selectbox("Tone", ["Professional", "Friendly", "Confident", "Concise"])
#     length = st.selectbox("Length", [
#         "Short (150-200 words)",
#         "Medium (250-350 words)",
#         "Long (400-500 words)"
#     ])
#     portfolio_options = [
#         "AI Customer Support & Lead Generation Chatbot (repumediaintelligence.com)",
#         "AI Medical Symptom Checker Chatbot (mobile app)",
#         "Global Premium Trades Inc - E-commerce (globalpremiumtrades-inc.com)",
#         "Global Premium Trades - E-commerce (globalpremiumtrades.com)",
#         "Al-Noor MDF Board Industries (alnoormdf.com)",
#         "RAG AI Knowledge Assistant for Agriculture (research project)",
#     ]
#     selected_portfolio = st.multiselect(
#         "Portfolio to Mention",
#         options=portfolio_options,
#         default=portfolio_options[:2]
#     )
#     opening_style = st.selectbox("Opening Style", [
#         "Start with client's problem",
#         "Start with a relevant result/achievement",
#         "Address client by name",
#         "Ask a smart question",
#     ])
#     top_k = st.slider("RAG chunks", 2, min(8, len(all_chunks)) if all_chunks else 8, 5)

# # ─────────────────────────────────────────────
# # GENERATE
# # ─────────────────────────────────────────────
# st.markdown("---")
# if st.button("🚀 Generate Proposal", type="primary", use_container_width=True):
#     if not job_title or not job_desc:
#         st.error("⚠️ Job Title aur Description zaroori hain!")
#         st.stop()

#     job = {
#         "title":       job_title,
#         "description": job_desc,
#         "client_name": client_name.strip(),
#         "budget":      budget.strip()
#     }
#     settings = {
#         "tone":               tone,
#         "length":             length,
#         "opening_style":      opening_style,
#         "selected_portfolio": ", ".join(selected_portfolio),
#     }

#     with st.spinner("🔍 Relevant sections retrieve ho rahe hain..."):
#         query = f"{job_title}. {job_desc[:500]}"
#         ctx, dbg = retrieve(query, embed_model, all_chunks, embeddings, top_k)

#     with st.spinner("✍️ Gemini proposal likh raha hai..."):
#         try:
#             proposal = generate_proposal(ctx, job, settings)
#         except Exception as e:
#             st.error(f"❌ Gemini error: {e}")
#             st.stop()

#     st.markdown("---")
#     st.subheader("📝 Generated Proposal")
#     st.text_area("Copy karo 👇", value=proposal, height=420)



import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer
import numpy as np
import re, os, warnings, logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROFILE_FILE   = "./profile.txt"
PROPOSALS_FILE = "./proposals.txt"

gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

st.set_page_config(page_title="Proposal Writer", page_icon="✍️", layout="wide")

# ─────────────────────────────────────────────
# EMBEDDING MODEL — ek baar load
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="📦 Embedding model load ho raha hai (sirf ek baar)...")
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# CHUNKS LOAD
# ─────────────────────────────────────────────
def load_profile_chunks(filepath: str) -> list[dict]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    parts = re.split(r"\[([A-Z_0-9]+)\]", raw)
    chunks = []
    for i in range(1, len(parts) - 1, 2):
        sec     = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            chunks.append({"section": sec, "content": f"[{sec}]\n{content}"})
    return chunks

def load_proposal_chunks(filepath: str) -> list[dict]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    parts = re.split(r"(Proposal-\d+:)", raw)
    chunks = []
    for i in range(1, len(parts) - 1, 2):
        label   = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            chunks.append({"section": label, "content": f"[EXAMPLE_PROPOSAL]\n{content}"})
    return chunks

# ─────────────────────────────────────────────
# IN-MEMORY INDEX — numpy cosine similarity
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Profile index ban raha hai...")
def build_index(_model):
    profile_chunks  = load_profile_chunks(PROFILE_FILE)
    proposal_chunks = load_proposal_chunks(PROPOSALS_FILE)
    all_chunks      = profile_chunks + proposal_chunks

    docs = [c["content"] for c in all_chunks]
    embeddings = _model.encode(docs, normalize_embeddings=True)  # shape: (N, 384)

    return all_chunks, embeddings

def retrieve(query: str, model, all_chunks, embeddings, top_k=5):
    q_emb = model.encode([query], normalize_embeddings=True)[0]  # (384,)

    # Cosine similarity = dot product (already normalized)
    scores = np.dot(embeddings, q_emb)  # (N,)
    top_indices = np.argsort(scores)[::-1][:top_k]

    ctx, dbg = [], []
    for idx in top_indices:
        ctx.append(all_chunks[idx]["content"])
        dbg.append({
            "section":   all_chunks[idx]["section"],
            "relevance": f"{round(float(scores[idx]) * 100, 1)}%"
        })
    return "\n\n---\n\n".join(ctx), dbg

# ─────────────────────────────────────────────
# PROPOSAL GENERATOR
# ─────────────────────────────────────────────
PORTFOLIO_LINKS = """
- AI Customer Support & Lead Generation Chatbot → repumediaintelligence.com
- AI Medical Symptom Checker Chatbot            → (mobile app, no public link)
- Global Premium Trades Inc - E-commerce        → globalpremiumtrades-inc.com
- Al-Noor MDF Board Industries - Website        → alnoormdf.com
- RAG AI Knowledge Assistant for Agriculture    → (research project, no public link)
""".strip()

def generate_proposal(ctx: str, job: dict, settings: dict) -> str:
    greeting = f"Hi {job['client_name']}," if job['client_name'] else "Hi,"

    portfolio_line = (
        "Mention ONLY these portfolio projects (with exact links where available): "
        + settings["selected_portfolio"]
    ) if settings["selected_portfolio"] else "Do not mention portfolio projects."

    budget_line = (
        f"Client budget is {job['budget']} — mention $25/hr rate naturally."
        if job["budget"] else "Do not mention hourly rate."
    )

    hook_guides = {
        "🏆 Proof of Results — show a similar win":
            'Open with a real, specific result from a past project similar to this job. Format: "I recently helped a client [specific outcome] in [timeframe]. I can do the same for you." Make it concrete — mention the result, not the effort.',
        "🤝 Relatable Experience — mirror their pain point":
            'Open by naming the exact pain point from the job description. Format: "I\'ve worked with clients who faced [their exact problem]. Here\'s how we solved it:" Then briefly show the solution. Client must feel you\'ve been here before.',
        "💪 Confidence + Guarantee — bold outcome claim":
            'Open with a confident, specific deliverable and timeframe. Format: "I can deliver [specific outcome] within [timeframe] — and I\'m confident enough to back it with [guarantee/revision policy]." Then give one quick proof point.',
        "🎁 Added Value — offer something extra":
            'Open by solving their core problem, then immediately offer something extra for free. Format: "Along with [main deliverable], I\'ll also [free bonus — e.g. code review, extra revision, deployment help]." Make the bonus feel relevant, not generic.',
        "📞 Helpful Approach — offer a no-pressure call":
            'Open by offering a free consultation call with zero commitment. Format: "I\'d love to jump on a quick call to walk you through exactly what\'s needed — no pressure at all. Even if we don\'t work together, you\'ll leave with clarity." Warm and confident tone.',
        "⚡ Straight to Proof — 3 quick wins":
            'Skip the intro. Go straight to 3 specific, relevant achievements or capabilities. Format: "I\'ll keep it short — [result/skill 1], [result/skill 2], [result/skill 3]." Each point must directly match something in the job description.',
        "🤔 Curiosity-Driven — question their requirement":
            'Open by referencing one specific detail from the job post that shows you read it carefully. Format: "I\'m curious what made you focus on [specific requirement from JD]. From what I see, you likely need [your smart insight about the real solution]." Show expertise through a smart observation.',
    }

    selected_hook = hook_guides.get(settings['opening_style'], "Start with the client's core problem and show you understand it deeply.")

    prompt = f"""You are an expert Upwork proposal writer. Write a SHORT, HIGH-CONVERTING, human-sounding proposal for freelancer Imran Riaz Chohan.

== FREELANCER PROFILE (retrieved) ==
{ctx}

== AVAILABLE PORTFOLIO LINKS ==
{PORTFOLIO_LINKS}

== JOB DETAILS ==
Title      : {job['title']}
Description: {job['description']}
Budget     : {job['budget'] or 'Not specified'}

== OPENING HOOK INSTRUCTION ==
{selected_hook}
This is the MOST important part. The first 2 sentences must stop the client from scrolling.
Use the client name "{job['client_name']}" in the opening if provided.

== WRITING INSTRUCTIONS ==
- Start with exactly: "{greeting}"
- Tone               : {settings['tone']}
- Target length      : {settings['length']}
- {portfolio_line}
- {budget_line}
- After the hook, briefly show relevant experience and tech match for THIS specific job
- Only mention skills that directly match the job description — no keyword dumping
- NO generic filler: "I am passionate", "I am a hard worker", "I would love to", "I am excited"
- NO bullet point lists — write in natural flowing paragraphs
- Vary sentence length — mix short punchy sentences with longer ones
- Sound like a real senior freelancer texting a colleague, not writing a cover letter
- End with ONE clear, natural call to action (call, demo, or quick question)
- Final 2 lines must be exactly:
Best Regards,
Imran Riaz

Write only the proposal. No commentary, no preamble, no subject line."""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return response.text

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
embed_model = load_embed_model()
all_chunks, embeddings = build_index(embed_model)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Imran Riaz Chohan")
    st.markdown("*AI Chatbot & Full Stack MERN Developer*")
    st.markdown("💰 **$25/hr** | ⭐ 5.0 | 🏆 100% JSS")
    st.markdown("---")
    st.success(f"✅ {len(all_chunks)} chunks indexed")
    st.markdown("**🔗 Portfolio**")
    st.markdown("[AI Support Chatbot](https://repumediaintelligence.com)")
    st.markdown("[Global Premium Trades](https://globalpremiumtrades-inc.com)")
    st.markdown("[Al-Noor MDF](https://alnoormdf.com)")

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.title("✍️ Upwork Proposal Writer")
st.caption("Job details bharo → RAG + Gemini → copy-paste ready proposal")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Job Details")
    job_title   = st.text_input("Job Title *", placeholder="e.g. React Developer for SaaS Platform")
    job_desc    = st.text_area("Job Description *", placeholder="Client ki full job posting yahan paste karo...", height=240)
    client_name = st.text_input("Client Name (optional)", placeholder="e.g. John, Sarah")
    budget      = st.text_input("Budget (optional)", placeholder="e.g. $500, $30/hr")

with col2:
    st.subheader("⚙️ Settings")
    tone = st.selectbox("Tone", ["Professional", "Friendly", "Confident", "Concise"])
    length = st.selectbox("Length", [
        "Short (150-200 words)",
        "Medium (250-350 words)",
        "Long (400-500 words)"
    ])
    portfolio_options = [
        "AI Customer Support & Lead Generation Chatbot (repumediaintelligence.com)",
        "AI Medical Symptom Checker Chatbot (mobile app)",
        "Global Premium Trades Inc - E-commerce (globalpremiumtrades-inc.com)",
        "Al-Noor MDF Board Industries (alnoormdf.com)",
        "RAG AI Knowledge Assistant for Agriculture (research project)",
    ]
    selected_portfolio = st.multiselect(
        "Portfolio to Mention",
        options=portfolio_options,
        default=portfolio_options[:2]
    )
    opening_style = st.selectbox("Opening Hook Style", [
        "🏆 Proof of Results — show a similar win",
        "🤝 Relatable Experience — mirror their pain point",
        "💪 Confidence + Guarantee — bold outcome claim",
        "🎁 Added Value — offer something extra",
        "📞 Helpful Approach — offer a no-pressure call",
        "⚡ Straight to Proof — 3 quick wins",
        "🤔 Curiosity-Driven — question their requirement",
    ])
    top_k = st.slider("RAG chunks", 2, min(8, len(all_chunks)) if all_chunks else 8, 5)

# ─────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────
st.markdown("---")
if st.button("🚀 Generate Proposal", type="primary", use_container_width=True):
    if not job_title or not job_desc:
        st.error("⚠️ Job Title aur Description zaroori hain!")
        st.stop()

    job = {
        "title":       job_title,
        "description": job_desc,
        "client_name": client_name.strip(),
        "budget":      budget.strip()
    }
    settings = {
        "tone":               tone,
        "length":             length,
        "opening_style":      opening_style,
        "selected_portfolio": ", ".join(selected_portfolio),
    }

    with st.spinner("🔍 Relevant sections retrieve ho rahe hain..."):
        query = f"{job_title}. {job_desc[:500]}"
        ctx, dbg = retrieve(query, embed_model, all_chunks, embeddings, top_k)

    with st.spinner("✍️ Gemini proposal likh raha hai..."):
        try:
            proposal = generate_proposal(ctx, job, settings)
        except Exception as e:
            st.error(f"❌ Gemini error: {e}")
            st.stop()

    st.markdown("---")
    st.subheader("📝 Generated Proposal")
    st.text_area("Copy karo 👇", value=proposal, height=420)