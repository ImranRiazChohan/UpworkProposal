# import streamlit as st
# from google import genai
# import chromadb
# from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
# from sentence_transformers import SentenceTransformer
# import re, os, hashlib
# import warnings
# warnings.filterwarnings("ignore")
# # ─────────────────────────────────────────────
# # CONFIG  — apni key aur file paths yahan
# # ─────────────────────────────────────────────
# GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# # GEMINI_API_KEY = "AIzaSyCL5tXdQxXrVZvxrmjjfyFVK-qZFEXwTgg1111"   # <-- apni key
# PROFILE_FILE   = "./profile.txt"
# PROPOSALS_FILE = "./proposals.txt"          # past proposals (optional, rakh lo same folder mein)
# CHROMA_DIR     = "./chroma_db"

# gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# st.set_page_config(page_title="Proposal Writer", page_icon="✍️", layout="wide")

# # ─────────────────────────────────────────────
# # EMBEDDING — ek baar load, hamesha cached
# # ─────────────────────────────────────────────
# @st.cache_resource(show_spinner="📦 Embedding model load ho raha hai (sirf ek baar)...")
# def load_embed_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# class LocalEmbedFn(EmbeddingFunction):
#     def __init__(self, model): self.model = model
#     def __call__(self, input: Documents) -> Embeddings:
#         return self.model.encode(list(input), normalize_embeddings=True).tolist()

# # ─────────────────────────────────────────────
# # PROFILE CHUNKS — [SECTION] based splitting
# # ─────────────────────────────────────────────
# def load_chunks(filepath: str) -> list[dict]:
#     with open(filepath, "r", encoding="utf-8") as f:
#         raw = f.read()
#     parts = re.split(r"\[([A-Z_0-9]+)\]", raw)
#     chunks = []
#     for i in range(1, len(parts) - 1, 2):
#         sec = parts[i].strip()
#         content = parts[i + 1].strip()
#         if content:
#             chunks.append({"id": sec.lower(), "section": sec, "content": f"[{sec}]\n{content}"})
#     return chunks

# def load_proposal_chunks(filepath: str) -> list[dict]:
#     """Past proposals ko numbered chunks mein load karo"""
#     if not os.path.exists(filepath):
#         return []
#     with open(filepath, "r", encoding="utf-8") as f:
#         raw = f.read()
#     # Split on Proposal-N: pattern
#     parts = re.split(r"(Proposal-\d+:)", raw)
#     chunks = []
#     for i in range(1, len(parts) - 1, 2):
#         label   = parts[i].strip().replace(":", "").lower().replace("-", "_")
#         content = parts[i + 1].strip()
#         if content:
#             chunks.append({"id": f"prop_{label}", "section": label.upper(), "content": f"[EXAMPLE_PROPOSAL]\n{content}"})
#     return chunks

# def file_hash(path: str) -> str:
#     with open(path, "rb") as f:
#         return hashlib.md5(f.read()).hexdigest()

# # ─────────────────────────────────────────────
# # CHROMADB — persistent, sirf tab rebuild ho
# # jab profile.txt ya proposals.txt change ho
# # ─────────────────────────────────────────────
# HASH_FILE = "./chroma_db/.content_hash"

# def get_stored_hash() -> str:
#     if os.path.exists(HASH_FILE):
#         with open(HASH_FILE) as f:
#             return f.read().strip()
#     return ""

# def save_hash(h: str):
#     os.makedirs(CHROMA_DIR, exist_ok=True)
#     with open(HASH_FILE, "w") as f:
#         f.write(h)

# @st.cache_resource(show_spinner="🔧 Profile index ban raha hai...")
# def build_collection(_embed_fn):
#     profile_chunks  = load_chunks(PROFILE_FILE)
#     proposal_chunks = load_proposal_chunks(PROPOSALS_FILE)
#     all_chunks      = profile_chunks + proposal_chunks

#     # Current content ka hash
#     h_parts = file_hash(PROFILE_FILE)
#     if os.path.exists(PROPOSALS_FILE):
#         h_parts += file_hash(PROPOSALS_FILE)
#     current_hash = hashlib.md5(h_parts.encode()).hexdigest()

#     chroma = chromadb.PersistentClient(path=CHROMA_DIR)

#     # Rebuild sirf tab karo jab content change hua ho
#     needs_rebuild = (get_stored_hash() != current_hash)

#     if needs_rebuild:
#         try:
#             chroma.delete_collection("upwork_profile")
#         except Exception:
#             pass

#     try:
#         col = chroma.get_collection("upwork_profile", embedding_function=_embed_fn)
#         if not needs_rebuild:
#             return col, all_chunks          # ← fast path, no rebuild
#     except Exception:
#         needs_rebuild = True

#     if needs_rebuild:
#         col = chroma.create_collection(
#             "upwork_profile",
#             embedding_function=_embed_fn,
#             metadata={"hnsw:space": "cosine"}
#         )
#         col.add(
#             ids=[c["id"] for c in all_chunks],
#             documents=[c["content"] for c in all_chunks],
#             metadatas=[{"section": c["section"]} for c in all_chunks]
#         )
#         save_hash(current_hash)

#     return col, all_chunks

# # ─────────────────────────────────────────────
# # RAG RETRIEVAL
# # ─────────────────────────────────────────────
# def retrieve(query: str, col, embed_fn, top_k=5) -> tuple[str, list]:
#     q_emb    = embed_fn([query])[0]
#     results  = col.query(query_embeddings=[q_emb], n_results=top_k,
#                          include=["documents", "distances", "metadatas"])
#     docs     = results["documents"][0]
#     dists    = results["distances"][0]
#     metas    = results["metadatas"][0]

#     ctx, dbg = [], []
#     for doc, dist, meta in zip(docs, dists, metas):
#         ctx.append(doc)
#         dbg.append({"section": meta["section"], "relevance": f"{round((1-dist)*100,1)}%"})
#     return "\n\n---\n\n".join(ctx), dbg

# # ─────────────────────────────────────────────
# # GEMINI PROPOSAL GENERATOR  — sharp prompt
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
#         if job["budget"] else
#         "Do not mention hourly rate."
#     )

#     prompt = f"""You are an expert Upwork proposal writer. Your job is to write a SHORT, HIGH-CONVERTING proposal for the freelancer Imran Riaz Chohan.

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
# - NO generic filler like "I am passionate about..." or "I am a hard worker"
# - Be specific, confident, and direct — write like a senior freelancer, not a beginner
# - Show you understand the client's problem in the first 2 sentences
# - End with a clear call to action (e.g. "Let's hop on a quick call" or "Happy to share a quick demo")
# - Final line must be exactly:
# Best Regards,
# Imran Riaz

# Write only the proposal text — no extra commentary."""

#     response = gemini_client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=prompt
#     )
#     return response.text

# # ─────────────────────────────────────────────
# # INIT
# # ─────────────────────────────────────────────
# embed_model = load_embed_model()
# embed_fn    = LocalEmbedFn(embed_model)

# chroma_col, all_chunks = None, []
# with st.sidebar:
#     st.markdown("## 👤 Imran Riaz Chohan")
#     st.markdown("*AI Chatbot & Full Stack MERN Developer*")
#     st.markdown("💰 **$25/hr** | ⭐ 5.0 | 🏆 100% JSS")
#     st.markdown("---")

#     if not os.path.exists(PROFILE_FILE):
#         st.error(f"❌ {PROFILE_FILE} nahi mila!")
#     else:
#         try:
#             chroma_col, all_chunks = build_collection(embed_fn)
#             st.success(f"✅ {len(all_chunks)} chunks indexed")
#         except Exception as e:
#             st.error(f"❌ {e}")

#     st.markdown("**🔗 Portfolio**")
#     st.markdown("[AI Support Chatbot](https://repumediaintelligence.com)")
#     st.markdown("[Global Premium Trades](https://globalpremiumtrades-inc.com)")
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

#     top_k = st.slider("RAG chunks", 2, min(8, max(len(all_chunks), 2)), 5,
#                       help="Kitne profile sections Gemini ko milenge")

# # ─────────────────────────────────────────────
# # GENERATE
# # ─────────────────────────────────────────────
# st.markdown("---")
# if st.button("🚀 Generate Proposal", type="primary", use_container_width=True):
#     if not job_title or not job_desc:
#         st.error("⚠️ Job Title aur Description zaroori hain!")
#         st.stop()
#     if chroma_col is None:
#         st.error("❌ ChromaDB ready nahi. Sidebar mein error dekho.")
#         st.stop()

#     job = {"title": job_title, "description": job_desc,
#            "client_name": client_name.strip(), "budget": budget.strip()}
#     settings = {
#         "tone":               tone,
#         "length":             length,
#         "opening_style":      opening_style,
#         "selected_portfolio": ", ".join(selected_portfolio),
#     }

#     with st.spinner("🔍 Relevant profile sections retrieve ho rahe hain..."):
#         query   = f"{job_title}. {job_desc[:500]}"
#         ctx, dbg = retrieve(query, chroma_col, embed_fn, top_k)

#     with st.spinner("✍️ Gemini proposal likh raha hai..."):
#         try:
#             proposal = generate_proposal(ctx, job, settings)
#         except Exception as e:
#             st.error(f"❌ Gemini error: {e}")
#             st.stop()

#     st.markdown("---")
#     st.subheader("📝 Generated Proposal")

#     # Copy-friendly text area
#     st.text_area("Proposal (select all → copy)", value=proposal, height=420, label_visibility="collapsed")
#     # st.markdown(proposal)         # rendered preview bhi




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

    prompt = f"""You are an expert Upwork proposal writer. Write a SHORT, HIGH-CONVERTING, human-sounding proposal for freelancer Imran Riaz Chohan.

== FREELANCER PROFILE (retrieved) ==
{ctx}

== AVAILABLE PORTFOLIO LINKS ==
{PORTFOLIO_LINKS}

== JOB DETAILS ==
Title      : {job['title']}
Description: {job['description']}
Budget     : {job['budget'] or 'Not specified'}

== WRITING INSTRUCTIONS ==
- Start with exactly: "{greeting}"
- Opening style      : {settings['opening_style']}
- Tone               : {settings['tone']}
- Target length      : {settings['length']}
- {portfolio_line}
- {budget_line}
- Match tech stack tightly to job requirements — only mention relevant skills
- NO generic filler like "I am passionate" or "I am a hard worker"
- Be specific, confident, direct — write like a senior freelancer
- First 2 sentences must show you understand the client's exact problem
- End with a clear, natural call to action
- Sound human — vary sentence length, avoid robotic patterns
- Final 2 lines must be exactly:
Best Regards,
Imran Riaz

Write only the proposal. No commentary, no preamble."""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
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

    with st.expander("🔍 RAG Debug"):
        for d in dbg:
            st.write(f"**{d['section']}** — {d['relevance']}")