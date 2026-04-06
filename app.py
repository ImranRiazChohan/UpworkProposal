# import streamlit as st
# from google import genai
# from google.genai import types
# import chromadb
# from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
# from sentence_transformers import SentenceTransformer
# import re
# import os

# # ─────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────
# GEMINI_API_KEY = "AIzaSyCL5tXdQxXrVZvxrmjjfyFVK-qZFEXwTgg"
# PROFILE_FILE   = "./profile.txt"
# CHROMA_DIR     = "./chroma_db"

# gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# st.set_page_config(page_title="Upwork Proposal Writer", page_icon="✍️", layout="wide")

# # ─────────────────────────────────────────────
# # STEP 1 — SentenceTransformer Embedding (FREE, LOCAL)
# # ─────────────────────────────────────────────
# @st.cache_resource(show_spinner="📦 Embedding model load ho raha hai...")
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# class LocalEmbeddingFunction(EmbeddingFunction):
#     def __init__(self, model):
#         self.model = model

#     def __call__(self, input: Documents) -> Embeddings:
#         embeddings = self.model.encode(list(input), normalize_embeddings=True)
#         return embeddings.tolist()

# # ─────────────────────────────────────────────
# # STEP 2 — profile.txt load aur chunk karo
# # ─────────────────────────────────────────────
# def load_profile_chunks(filepath: str) -> list[dict]:
#     with open(filepath, "r", encoding="utf-8") as f:
#         raw = f.read()

#     pattern = r"\[([A-Z_0-9]+)\]"
#     parts   = re.split(pattern, raw)

#     chunks = []
#     for i in range(1, len(parts) - 1, 2):
#         section_id = parts[i].strip()
#         content    = parts[i + 1].strip()
#         if content:
#             chunks.append({
#                 "id":      section_id.lower(),
#                 "section": section_id,
#                 "content": f"[{section_id}]\n{content}"
#             })
#     return chunks

# # ─────────────────────────────────────────────
# # STEP 3 — ChromaDB persistent collection
# # ─────────────────────────────────────────────
# @st.cache_resource(show_spinner="🔧 profile.txt ChromaDB mein index ho raha hai...")
# def build_chroma_collection(_embed_fn):
#     chunks = load_profile_chunks(PROFILE_FILE)

#     chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

#     try:
#         chroma_client.delete_collection("upwork_profile")
#     except Exception:
#         pass

#     collection = chroma_client.create_collection(
#         name="upwork_profile",
#         embedding_function=_embed_fn,
#         metadata={"hnsw:space": "cosine"}
#     )

#     collection.add(
#         ids=[c["id"] for c in chunks],
#         documents=[c["content"] for c in chunks],
#         metadatas=[{"section": c["section"]} for c in chunks]
#     )

#     return collection, chunks

# # ─────────────────────────────────────────────
# # STEP 4 — RAG Retrieval
# # ─────────────────────────────────────────────
# def retrieve_context(query: str, collection, embed_fn, top_k: int = 4) -> tuple[str, list]:
#     query_embedding = embed_fn([query])[0]

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k,
#         include=["documents", "distances", "metadatas"]
#     )

#     docs      = results["documents"][0]
#     distances = results["distances"][0]
#     metas     = results["metadatas"][0]

#     context_parts = []
#     debug_info    = []
#     for doc, dist, meta in zip(docs, distances, metas):
#         similarity = round((1 - dist) * 100, 1)
#         context_parts.append(doc)
#         debug_info.append({"section": meta["section"], "relevance": f"{similarity}%"})

#     return "\n\n---\n\n".join(context_parts), debug_info

# # ─────────────────────────────────────────────
# # STEP 5 — Gemini Proposal Generator
# # ─────────────────────────────────────────────
# def generate_proposal(context: str, job: dict, settings: dict) -> str:
#     greeting = f"Hi {job['client_name']}," if job['client_name'] else "Hi,"

#     prompt = f"""You are an expert Upwork proposal writer. Write a winning, human-sounding proposal.

# === FREELANCER PROFILE (RAG Retrieved) ===
# {context}

# === JOB DETAILS ===
# Title: {job['title']}
# Description: {job['description']}
# Client Name: {job['client_name'] or 'Not provided'}
# Budget: {job['budget'] or 'Not specified'}

# === SETTINGS ===
# Tone: {settings['tone']}
# Length: {settings['length']}
# Opening Style: {settings['opening_style']}
# Portfolio to Mention: {settings['selected_portfolio']}

# === PORTFOLIO LINKS (use these exactly) ===
# - AI Customer Support & Lead Generation Chatbot: repumediaintelligence.com
# - AI Medical Symptom Checker Chatbot: (mobile app, no public link)
# - Global Premium Trades Inc - E-commerce Marketplace: globalpremiumtrades-inc.com
# - Global Premium Trades - E-commerce Marketplace: globalpremiumtrades.com
# - Al-Noor MDF Board Industries - Company Website: alnoormdf.com

# === RULES ===
# 1. Start with: "{greeting}"
# 2. Use opening style: {settings['opening_style']}
# 3. Tone: {settings['tone']}
# 4. Mention ONLY the selected portfolio projects with their exact links(Mandatory)
# 5. Match tech stack to job requirements
# 6. Mention $25/hr rate only if client mentioned budget
# 7. End with a strong call to action
# 8. Target length: {settings['length']}
# 9. Write real content — no placeholders
# 10. Sound human, confident, and specific to this job
# 11. Always end with exactly:
# Best Regards,
# Imran Riaz

# Write the proposal:"""

#     response = gemini_client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=prompt
#     )
#     return response.text


# # ─────────────────────────────────────────────
# # INIT — Models & Collection
# # ─────────────────────────────────────────────
# embed_model  = load_embedding_model()
# embed_fn     = LocalEmbeddingFunction(embed_model)

# chroma_collection = None
# all_chunks        = []

# # ─────────────────────────────────────────────
# # SIDEBAR
# # ─────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 👤 Imran Riaz Chohan")
#     st.markdown("*AI Chatbot & Full Stack MERN Developer*")
#     st.markdown("💰 **$25.00/hr** | ⭐ 5.0 (3 reviews)")
#     st.markdown("---")

#     if not os.path.exists(PROFILE_FILE):
#         st.error(f"❌ {PROFILE_FILE} nahi mila! Same folder mein rakho.")
#     else:
#         try:
#             chroma_collection, all_chunks = build_chroma_collection(embed_fn)
#         except Exception as e:
#             st.error(f"❌ Error: {e}")


#     st.markdown("**🔗 Portfolio**")
#     st.markdown("[AI Support Chatbot](https://repumediaintelligence.com)")
#     st.markdown("[Global Premium Trades](https://globalpremiumtrades-inc.com)")
#     st.markdown("[Al-Noor MDF](https://alnoormdf.com)")

# # ─────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────
# st.title("✍️ Upwork Proposal Writer")
# st.markdown("Job details bharo — local embeddings + Gemini se winning proposal generate hogi")
# st.markdown("---")

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.subheader("📋 Job Details")
#     job_title       = st.text_input("Job Title *", placeholder="e.g. React Developer for SaaS Platform")
#     job_description = st.text_area("Job Description *", placeholder="Client ki full job posting yahan paste karo...", height=220)
#     client_name     = st.text_input("Client Name (optional)", placeholder="e.g. John, Sarah")
#     budget          = st.text_input("Budget (optional)", placeholder="e.g. $500, $30/hr")

# with col2:
#     st.subheader("⚙️ Settings")

#     tone = st.selectbox("Proposal Tone", ["Professional", "Friendly", "Confident", "Concise"])

#     proposal_length = st.selectbox("Length", [
#         "Short (150-200 words)",
#         "Medium (250-350 words)",
#         "Long (400-500 words)"
#     ])

#     portfolio_options = [
#         "AI Customer Support & Lead Generation Chatbot",
#         "AI Medical Symptom Checker Chatbot",
#         "Global Premium Trades Inc - E-commerce Marketplace",
#         "Global Premium Trades - E-commerce Marketplace",
#         "Al-Noor MDF Board Industries - Company Website",
#     ]
#     include_portfolio = st.multiselect(
#         "Portfolio to Mention",
#         options=portfolio_options,
#         default=portfolio_options[:2]
#     )

#     opening_style = st.selectbox("Opening Style", [
#         "Address client by name",
#         "Start with their problem",
#         "Start with your result/achievement",
#         "Ask a smart question"
#     ])

#     top_k = st.slider(
#         "RAG chunks to retrieve",
#         min_value=2,
#         max_value=max(4, len(all_chunks)) if all_chunks else 9,
#         value=4,
#         help="Kitne profile.txt sections Gemini ko milenge"
#     )

# # ─────────────────────────────────────────────
# # GENERATE
# # ─────────────────────────────────────────────
# st.markdown("---")
# generate_btn = st.button("🚀 Generate Proposal", type="primary", use_container_width=True)

# if generate_btn:
#     if not job_title or not job_description:
#         st.error("⚠️ Job Title aur Description zaroori hain!")
#         st.stop()
#     if chroma_collection is None:
#         st.error("❌ ChromaDB ready nahi. Sidebar mein error dekho.")
#         st.stop()

#     job = {
#         "title":       job_title,
#         "description": job_description,
#         "client_name": client_name.strip(),
#         "budget":      budget.strip()
#     }
#     settings = {
#         "tone":               tone,
#         "length":             proposal_length,
#         "opening_style":      opening_style,
#         "selected_portfolio": ", ".join(include_portfolio) if include_portfolio else "All projects"
#     }

#     with st.spinner("🔍 Profile se relevant chunks retrieve ho rahe hain..."):
#         query = f"{job_title}. {job_description[:400]}"
#         retrieved_context, debug_info = retrieve_context(query, chroma_collection, embed_fn, top_k)

#     with st.spinner("✍️ Gemini proposal likh raha hai..."):
#         try:
#             proposal = generate_proposal(retrieved_context, job, settings)
#         except Exception as e:
#             st.error(f"❌ Gemini error: {e}")
#             st.stop()

#     st.markdown("---")
#     st.subheader("📝 Generated Proposal")
#     st.markdown(proposal)




import streamlit as st
from google import genai
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
import re, os, hashlib
import warnings
warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────
# CONFIG  — apni key aur file paths yahan
# ─────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyCL5tXdQxXrVZvxrmjjfyFVK-qZFEXwTgg"   # <-- apni key
PROFILE_FILE   = "./profile.txt"
PROPOSALS_FILE = "./proposals.txt"          # past proposals (optional, rakh lo same folder mein)
CHROMA_DIR     = "./chroma_db"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="Proposal Writer", page_icon="✍️", layout="wide")

# ─────────────────────────────────────────────
# EMBEDDING — ek baar load, hamesha cached
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="📦 Embedding model load ho raha hai (sirf ek baar)...")
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

class LocalEmbedFn(EmbeddingFunction):
    def __init__(self, model): self.model = model
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(list(input), normalize_embeddings=True).tolist()

# ─────────────────────────────────────────────
# PROFILE CHUNKS — [SECTION] based splitting
# ─────────────────────────────────────────────
def load_chunks(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    parts = re.split(r"\[([A-Z_0-9]+)\]", raw)
    chunks = []
    for i in range(1, len(parts) - 1, 2):
        sec = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            chunks.append({"id": sec.lower(), "section": sec, "content": f"[{sec}]\n{content}"})
    return chunks

def load_proposal_chunks(filepath: str) -> list[dict]:
    """Past proposals ko numbered chunks mein load karo"""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    # Split on Proposal-N: pattern
    parts = re.split(r"(Proposal-\d+:)", raw)
    chunks = []
    for i in range(1, len(parts) - 1, 2):
        label   = parts[i].strip().replace(":", "").lower().replace("-", "_")
        content = parts[i + 1].strip()
        if content:
            chunks.append({"id": f"prop_{label}", "section": label.upper(), "content": f"[EXAMPLE_PROPOSAL]\n{content}"})
    return chunks

def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# ─────────────────────────────────────────────
# CHROMADB — persistent, sirf tab rebuild ho
# jab profile.txt ya proposals.txt change ho
# ─────────────────────────────────────────────
HASH_FILE = "./chroma_db/.content_hash"

def get_stored_hash() -> str:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            return f.read().strip()
    return ""

def save_hash(h: str):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(HASH_FILE, "w") as f:
        f.write(h)

@st.cache_resource(show_spinner="🔧 Profile index ban raha hai...")
def build_collection(_embed_fn):
    profile_chunks  = load_chunks(PROFILE_FILE)
    proposal_chunks = load_proposal_chunks(PROPOSALS_FILE)
    all_chunks      = profile_chunks + proposal_chunks

    # Current content ka hash
    h_parts = file_hash(PROFILE_FILE)
    if os.path.exists(PROPOSALS_FILE):
        h_parts += file_hash(PROPOSALS_FILE)
    current_hash = hashlib.md5(h_parts.encode()).hexdigest()

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)

    # Rebuild sirf tab karo jab content change hua ho
    needs_rebuild = (get_stored_hash() != current_hash)

    if needs_rebuild:
        try:
            chroma.delete_collection("upwork_profile")
        except Exception:
            pass

    try:
        col = chroma.get_collection("upwork_profile", embedding_function=_embed_fn)
        if not needs_rebuild:
            return col, all_chunks          # ← fast path, no rebuild
    except Exception:
        needs_rebuild = True

    if needs_rebuild:
        col = chroma.create_collection(
            "upwork_profile",
            embedding_function=_embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=[c["id"] for c in all_chunks],
            documents=[c["content"] for c in all_chunks],
            metadatas=[{"section": c["section"]} for c in all_chunks]
        )
        save_hash(current_hash)

    return col, all_chunks

# ─────────────────────────────────────────────
# RAG RETRIEVAL
# ─────────────────────────────────────────────
def retrieve(query: str, col, embed_fn, top_k=5) -> tuple[str, list]:
    q_emb    = embed_fn([query])[0]
    results  = col.query(query_embeddings=[q_emb], n_results=top_k,
                         include=["documents", "distances", "metadatas"])
    docs     = results["documents"][0]
    dists    = results["distances"][0]
    metas    = results["metadatas"][0]

    ctx, dbg = [], []
    for doc, dist, meta in zip(docs, dists, metas):
        ctx.append(doc)
        dbg.append({"section": meta["section"], "relevance": f"{round((1-dist)*100,1)}%"})
    return "\n\n---\n\n".join(ctx), dbg

# ─────────────────────────────────────────────
# GEMINI PROPOSAL GENERATOR  — sharp prompt
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
        if job["budget"] else
        "Do not mention hourly rate."
    )

    prompt = f"""You are an expert Upwork proposal writer. Your job is to write a SHORT, HIGH-CONVERTING proposal for the freelancer Imran Riaz Chohan.

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
- NO generic filler like "I am passionate about..." or "I am a hard worker"
- Be specific, confident, and direct — write like a senior freelancer, not a beginner
- Show you understand the client's problem in the first 2 sentences
- End with a clear call to action (e.g. "Let's hop on a quick call" or "Happy to share a quick demo")
- Final line must be exactly:
Best Regards,
Imran Riaz

Write only the proposal text — no extra commentary."""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
embed_model = load_embed_model()
embed_fn    = LocalEmbedFn(embed_model)

chroma_col, all_chunks = None, []
with st.sidebar:
    st.markdown("## 👤 Imran Riaz Chohan")
    st.markdown("*AI Chatbot & Full Stack MERN Developer*")
    st.markdown("💰 **$25/hr** | ⭐ 5.0 | 🏆 100% JSS")
    st.markdown("---")

    if not os.path.exists(PROFILE_FILE):
        st.error(f"❌ {PROFILE_FILE} nahi mila!")
    else:
        try:
            chroma_col, all_chunks = build_collection(embed_fn)
            st.success(f"✅ {len(all_chunks)} chunks indexed")
        except Exception as e:
            st.error(f"❌ {e}")

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

    top_k = st.slider("RAG chunks", 2, min(8, max(len(all_chunks), 2)), 5,
                      help="Kitne profile sections Gemini ko milenge")

# ─────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────
st.markdown("---")
if st.button("🚀 Generate Proposal", type="primary", use_container_width=True):
    if not job_title or not job_desc:
        st.error("⚠️ Job Title aur Description zaroori hain!")
        st.stop()
    if chroma_col is None:
        st.error("❌ ChromaDB ready nahi. Sidebar mein error dekho.")
        st.stop()

    job = {"title": job_title, "description": job_desc,
           "client_name": client_name.strip(), "budget": budget.strip()}
    settings = {
        "tone":               tone,
        "length":             length,
        "opening_style":      opening_style,
        "selected_portfolio": ", ".join(selected_portfolio),
    }

    with st.spinner("🔍 Relevant profile sections retrieve ho rahe hain..."):
        query   = f"{job_title}. {job_desc[:500]}"
        ctx, dbg = retrieve(query, chroma_col, embed_fn, top_k)

    with st.spinner("✍️ Gemini proposal likh raha hai..."):
        try:
            proposal = generate_proposal(ctx, job, settings)
        except Exception as e:
            st.error(f"❌ Gemini error: {e}")
            st.stop()

    st.markdown("---")
    st.subheader("📝 Generated Proposal")

    # Copy-friendly text area
    st.text_area("Proposal (select all → copy)", value=proposal, height=420, label_visibility="collapsed")
    # st.markdown(proposal)         # rendered preview bhi