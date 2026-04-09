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
# EMBEDDING MODEL
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
# IN-MEMORY INDEX
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Profile index ban raha hai...")
def build_index(_model):
    profile_chunks  = load_profile_chunks(PROFILE_FILE)
    proposal_chunks = load_proposal_chunks(PROPOSALS_FILE)
    all_chunks      = profile_chunks + proposal_chunks
    docs            = [c["content"] for c in all_chunks]
    embeddings      = _model.encode(docs, normalize_embeddings=True)
    return all_chunks, embeddings

def retrieve(query: str, model, all_chunks, embeddings, top_k=5):
    q_emb   = model.encode([query], normalize_embeddings=True)[0]
    scores  = np.dot(embeddings, q_emb)
    top_idx = np.argsort(scores)[::-1][:top_k]
    ctx, dbg = [], []
    for idx in top_idx:
        ctx.append(all_chunks[idx]["content"])
        dbg.append({
            "section":   all_chunks[idx]["section"],
            "relevance": f"{round(float(scores[idx]) * 100, 1)}%"
        })
    return "\n\n---\n\n".join(ctx), dbg

# ─────────────────────────────────────────────
# AUTO HOOK SELECTOR
# ─────────────────────────────────────────────
HOOKS = [
    {
        "name": "Straight to Proof",
        "keywords": ["chatbot", "rag", "llm", "ai", "openai", "production", "live", "pipeline", "deployed", "bot"],
        "template": "I've already built this — here's the live proof:",
        "best_for": "AI/chatbot/RAG jobs — show live project instantly"
    },
    {
        "name": "Relatable Experience",
        "keywords": ["problem", "issue", "struggling", "fix", "improve", "optimize", "broken", "slow", "bug", "error"],
        "template": "I've solved this exact problem before — here's how:",
        "best_for": "Jobs describing a specific pain point or bug"
    },
    {
        "name": "Proof of Results",
        "keywords": ["result", "outcome", "goal", "increase", "reduce", "automate", "save", "scale", "grow", "revenue"],
        "template": "I recently helped a client achieve a similar outcome — faster than expected.",
        "best_for": "Jobs focused on business outcomes or metrics"
    },
    {
        "name": "Confidence + Guarantee",
        "keywords": ["build", "create", "develop", "deploy", "launch", "mvp", "deliver", "deadline", "complete", "finish"],
        "template": "I can deliver exactly this — clean code, on time, no guesswork.",
        "best_for": "Fixed-scope build/deploy jobs"
    },
    {
        "name": "Curiosity-Driven",
        "keywords": ["custom", "unique", "specific", "niche", "complex", "particular", "special", "unusual", "tailored"],
        "template": "Your requirement is specific — and that's exactly the kind of challenge I enjoy.",
        "best_for": "Niche or complex jobs with unique requirements"
    },
    {
        "name": "Helpful Approach",
        "keywords": ["maintain", "support", "ongoing", "long-term", "update", "retainer", "manage", "help", "assist"],
        "template": "I'd love to jump on a quick call to walk you through how I'd approach this — no pressure.",
        "best_for": "Ongoing/maintenance/support jobs"
    },
    {
        "name": "Added Value",
        "keywords": ["full stack", "mern", "react", "node", "website", "web app", "frontend", "backend", "api", "ecommerce"],
        "template": "Beyond solving your core requirement, I'll make sure the codebase is clean, documented, and easy to scale.",
        "best_for": "Full stack web/MERN jobs"
    },
]

def auto_select_hook(job_title: str, job_desc: str, client_name: str = "") -> dict:
    text   = (job_title + " " + job_desc).lower()
    scores = [(sum(1 for kw in h["keywords"] if kw in text), h) for h in HOOKS]
    scores.sort(key=lambda x: x[0], reverse=True)
    best   = scores[0][1]
    name   = client_name.strip() if client_name.strip() else ""
    greeting = f"Hi {name}," if name else "Hi,"
    return {**best, "greeting": greeting}

# ─────────────────────────────────────────────
# PORTFOLIO LINKS
# ─────────────────────────────────────────────
PORTFOLIO_LINKS = """- AI Customer Support & Lead Generation Chatbot → repumediaintelligence.com
- Global Premium Trades Inc - E-commerce Marketplace → globalpremiumtrades-inc.com
- Al-Noor MDF Board Industries - Website → alnoormdf.com
- AI Medical Symptom Checker Chatbot → (mobile app, OpenAI + Firebase)
- RAG AI Knowledge Assistant for Agriculture → (research project, YOLOv8 + LLama3)"""

# ─────────────────────────────────────────────
# PROPOSAL GENERATOR
# ─────────────────────────────────────────────
def generate_proposal(ctx: str, job: dict, settings: dict, hook: dict) -> str:

    budget_line = (
        f"Client budget is {job['budget']} — mention $25/hr rate naturally."
        if job["budget"] else "Do not mention hourly rate."
    )

    portfolio_line = (
        "Mention ONLY these selected portfolio projects: " + settings["selected_portfolio"]
    ) if settings["selected_portfolio"] else "Pick 1-2 most relevant portfolio projects from the list."

    prompt = f"""You are an expert Upwork proposal writer. Your ONLY goal: write a proposal that gets OPENED and VIEWED by the client.

Study these winning proposals carefully — match their style, directness, and confidence:
{ctx}

== AVAILABLE PORTFOLIO LINKS ==
{PORTFOLIO_LINKS}

== JOB DETAILS ==
Title      : {job['title']}
Description: {job['description']}
Budget     : {job['budget'] or 'Not specified'}

== AUTO-SELECTED HOOK ==
Hook Type : {hook['name']}
Hook Reason: {hook['best_for']}

== STRICT WRITING RULES ==
1. Start with exactly: "{hook['greeting']}"
2. Second line: use this hook opener naturally → "{hook['template']}"
3. Third and fourth lines: immediately show 1-2 LIVE portfolio links with ONE line context each
   Example format:
   → repumediaintelligence.com — AI chatbot with lead capture + CRM, live in production
   → globalpremiumtrades-inc.com — full MERN e-commerce, dynamic listings + responsive UI
4. After portfolio links: show you DEEPLY understand the client's exact problem (mirror their words)
5. Briefly explain HOW you would solve it — be specific, not generic
6. Max 3-4 bullet points for deliverables or proof — keep them tight
7. {portfolio_line}
8. {budget_line}
9. End with ONE specific question about their project — makes them want to reply
10. Tone: {settings['tone']}
11. Length: {settings['length']}
12. NO filler lines: "I am passionate", "I am a hard worker", "I would love to", "I am confident"
13. Sound like a senior freelancer — direct, specific, zero fluff
14. Last 2 lines must be exactly:
Best Regards,
Imran Riaz

Write ONLY the proposal. No commentary, no preamble, no explanation."""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return response.text

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
embed_model            = load_embed_model()
all_chunks, embeddings = build_index(embed_model)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Imran Riaz Chohan")
    st.markdown("*AI Chatbot & Full Stack MERN Developer*")
    st.markdown("💰 **$25/hr** | ⭐ 5.0 | 🏆 100% JSS")
    st.markdown("---")
    st.markdown("**🔗 Portfolio**")
    st.markdown("[AI Support Chatbot](https://repumediaintelligence.com)")
    st.markdown("[Global Premium Trades-inc](https://globalpremiumtrades-inc.com)")
    st.markdown("[Global Premium Trades](https://globalpremiumtrades.com)")
    st.markdown("[Al-Noor MDF](https://alnoormdf.com)")
    st.markdown("---")
    st.caption(f"📚 {len(all_chunks)} chunks indexed")

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.title("✍️ Upwork Proposal Writer")
st.caption("Job paste karo → hook auto-select → RAG + Gemini → view-worthy proposal")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Job Details")
    job_title   = st.text_input("Job Title *", placeholder="e.g. React Developer for SaaS Platform")
    job_desc    = st.text_area("Job Description *", placeholder="Client ki full job posting yahan paste karo...", height=240)
    client_name = st.text_input("Client Name (optional)", placeholder="e.g. John, Sarah")
    budget      = st.text_input("Budget (optional)", placeholder="e.g. $500, $30/hr")

    # ── Live Hook Preview ──
    if job_title and job_desc:
        hook_preview = auto_select_hook(job_title, job_desc, client_name)
        st.markdown("---")
        col_h1, col_h2 = st.columns([1, 2])
        with col_h1:
            st.markdown(f"**🪝 Auto Hook:**")
            st.success(f"`{hook_preview['name']}`")
        with col_h2:
            st.markdown(f"**Opening line:**")
            st.info(f"{hook_preview['greeting']} {hook_preview['template']}")
        st.caption(f"Reason: {hook_preview['best_for']}")

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
        "Global Premium Trades - E-commerce (globalpremiumtrades.com)",
        "Al-Noor MDF Board Industries (alnoormdf.com)",
        "RAG AI Knowledge Assistant for Agriculture (research project)",
    ]
    selected_portfolio = st.multiselect(
        "Portfolio to Mention",
        options=portfolio_options,
        default=portfolio_options[:2]
    )
    opening_style = st.selectbox("Opening Style", [
        "Start with client's problem",
        "Start with a relevant result/achievement",
        "Address client by name",
        "Ask a smart question",
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

    hook = auto_select_hook(job_title, job_desc, client_name)

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
        query    = f"{job_title}. {job_desc[:500]}"
        ctx, dbg = retrieve(query, embed_model, all_chunks, embeddings, top_k)

    with st.spinner("✍️ Gemini proposal likh raha hai..."):
        try:
            proposal = generate_proposal(ctx, job, settings, hook)
        except Exception as e:
            st.error(f"❌ Gemini error: {e}")
            st.stop()

    st.markdown("---")

    # Hook used badge
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader("📝 Generated Proposal")
    with c2:
        st.markdown(f"<br><span style='background:#e8f4e8;color:#2d6a2d;padding:4px 10px;border-radius:6px;font-size:12px'>🪝 {hook['name']}</span>", unsafe_allow_html=True)

    st.text_area("Copy karo 👇", value=proposal, height=420)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "📥 Download (.txt)",
            data=proposal,
            file_name=f"proposal_{job_title[:25].replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col_b:
        st.metric("Word Count", len(proposal.split()))

    with st.expander("🔍 RAG Debug — Retrieved Chunks"):
        for d in dbg:
            st.caption(f"✅ {d['section']} — {d['relevance']}")