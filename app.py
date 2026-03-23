# app.py
# HireIQ - AI Recruitment Intelligence Platform
# The World's First "Talent DNA Matching" System
# Run with: streamlit run app.py

import streamlit as st
import chromadb
import os
import time
import tempfile
import json
import re
import pandas as pd
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="HireIQ — Talent Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CORE COMPONENTS
# ─────────────────────────────────────────

@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.2", temperature=0.1)

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

@st.cache_resource
def get_vector_db():
    return chromadb.PersistentClient(path="./hireiq_data")

# ─────────────────────────────────────────
# STYLING — PREMIUM HR DASHBOARD
# ─────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* ── Main Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
        padding: 35px 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero-banner h1 {
        font-size: 2.8em;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-banner .tagline {
        font-size: 1.1em;
        opacity: 0.9;
        margin-top: 8px;
        font-weight: 300;
    }
    .hero-banner .badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin-top: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }

    /* ── Candidate DNA Card ── */
    .dna-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .dna-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 4px 0 0 4px;
    }
    .dna-card:hover {
        border-color: rgba(102,126,234,0.7);
        box-shadow: 0 8px 30px rgba(102,126,234,0.2);
        transform: translateY(-2px);
    }

    /* ── Match Score Ring ── */
    .score-ring-wrapper {
        text-align: center;
        padding: 15px;
    }
    .score-excellent { color: #00ff88; font-size: 2.5em; font-weight: 700; }
    .score-good { color: #ffd700; font-size: 2.5em; font-weight: 700; }
    .score-average { color: #ff8c00; font-size: 2.5em; font-weight: 700; }
    .score-poor { color: #ff4757; font-size: 2.5em; font-weight: 700; }

    /* ── Talent Badge ── */
    .talent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin: 3px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-skill {
        background: rgba(102,126,234,0.2);
        color: #a78bfa;
        border: 1px solid rgba(102,126,234,0.4);
    }
    .badge-match {
        background: rgba(0,255,136,0.1);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.3);
    }
    .badge-gap {
        background: rgba(255,71,87,0.1);
        color: #ff6b7a;
        border: 1px solid rgba(255,71,87,0.3);
    }
    .badge-neutral {
        background: rgba(255,215,0,0.1);
        color: #ffd700;
        border: 1px solid rgba(255,215,0,0.3);
    }

    /* ── Contact Card ── */
    .contact-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1));
        border: 1px solid rgba(102,126,234,0.25);
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
    }
    .contact-item {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 6px 0;
        color: #c8c8f0;
        font-size: 13px;
    }

    /* ── Rank Badge ── */
    .rank-1 {
        background: linear-gradient(135deg, #ffd700, #ff8c00);
        color: #000;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }
    .rank-2 {
        background: linear-gradient(135deg, #c0c0c0, #a0a0a0);
        color: #000;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }
    .rank-3 {
        background: linear-gradient(135deg, #cd7f32, #a0522d);
        color: #fff;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }
    .rank-other {
        background: rgba(255,255,255,0.1);
        color: #ccc;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }

    /* ── Progress Bar ── */
    .match-bar-wrapper {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        height: 8px;
        margin: 8px 0;
        overflow: hidden;
    }
    .match-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    /* ── JD Box ── */
    .jd-input-box {
        background: rgba(102,126,234,0.05);
        border: 2px dashed rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }

    /* ── Stats Card ── */
    .stats-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stats-number {
        font-size: 2.2em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stats-label {
        color: #888;
        font-size: 12px;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Section Header ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(102,126,234,0.2);
    }
    .section-header h3 {
        color: #a78bfa;
        margin: 0;
        font-size: 1.1em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a3e);
        border-right: 1px solid rgba(102,126,234,0.2);
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
    }

    /* ── Tab override ── */
    .stTabs [data-baseweb="tab"] {
        color: #888 !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #a78bfa !important;
        border-bottom-color: #667eea !important;
    }

    /* ── Text colors ── */
    h1, h2, h3, h4 { color: #e8e8ff !important; }
    p, li { color: #c0c0d0 !important; }
    label { color: #a0a0c0 !important; }
    .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        color: #e0e0ff !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DOCUMENT LOADING
# ─────────────────────────────────────────

def load_resume(uploaded_file):
    """Load resume from PDF, DOCX, or TXT"""
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{uploaded_file.name.split('.')[-1]}"
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    ext = uploaded_file.name.split('.')[-1].lower()

    try:
        if ext == 'pdf':
            loader = PyPDFLoader(tmp_path)
        elif ext in ['doc', 'docx']:
            loader = Docx2txtLoader(tmp_path)
        elif ext in ['txt', 'md']:
            loader = TextLoader(tmp_path)
        else:
            st.error(f"❌ Unsupported: {ext}")
            return None

        docs = loader.load()
        os.unlink(tmp_path)
        return docs

    except Exception as e:
        st.error(f"❌ Load error: {e}")
        return None


# ─────────────────────────────────────────
# CANDIDATE EXTRACTION ENGINE
# ─────────────────────────────────────────

def extract_candidate_profile(raw_text: str, filename: str) -> dict:
    """
    Uses LLM to extract structured candidate data
    from raw resume text.
    Returns a rich candidate profile dict.
    """
    llm = get_llm()

    prompt = f"""You are a world-class HR data extraction engine.
Extract ALL candidate information from this resume.
Return ONLY a valid JSON object. No extra text.

EXTRACT THESE FIELDS:
{{
  "full_name": "candidate full name or 'Unknown'",
  "email": "email address or 'Not found'",
  "phone": "phone number or 'Not found'",
  "location": "city, country or 'Not found'",
  "linkedin": "linkedin URL or 'Not found'",
  "github": "github URL or 'Not found'",
  "portfolio": "portfolio/website URL or 'Not found'",
  "current_role": "current or most recent job title",
  "total_experience_years": "number like 3 or 0",
  "education": "highest degree + field + institution",
  "skills": ["skill1", "skill2", "skill3"],
  "tools": ["tool1", "tool2"],
  "certifications": ["cert1", "cert2"],
  "languages": ["English", "Spanish"],
  "summary": "2-3 sentence professional summary",
  "companies": ["company1", "company2"],
  "achievements": ["achievement1", "achievement2"]
}}

RESUME TEXT:
{raw_text[:3000]}

JSON OUTPUT:"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Clean JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        profile = json.loads(content.strip())
        profile["source_file"] = filename
        profile["indexed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return profile

    except Exception as e:
        return {
            "full_name": filename.replace(".pdf","").replace(".docx",""),
            "email": "Parse error",
            "phone": "Parse error",
            "location": "Unknown",
            "linkedin": "Not found",
            "github": "Not found",
            "portfolio": "Not found",
            "current_role": "Unknown",
            "total_experience_years": 0,
            "education": "Unknown",
            "skills": [],
            "tools": [],
            "certifications": [],
            "languages": [],
            "summary": f"Could not parse profile: {e}",
            "companies": [],
            "achievements": [],
            "source_file": filename,
            "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }


# ─────────────────────────────────────────
# VECTOR STORE — RESUME INDEXING
# ─────────────────────────────────────────

def index_resume(docs, filename: str, org_id: str) -> dict:
    """
    1. Extract full text from docs
    2. Use LLM to extract structured profile
    3. Store embeddings + metadata in ChromaDB
    4. Return the extracted profile
    """
    full_text = " ".join([d.page_content for d in docs])

    # Extract structured profile via LLM
    profile = extract_candidate_profile(full_text, filename)

    # Chunk the resume
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80
    )
    chunks = splitter.split_documents(docs)

    db = get_vector_db()
    collection = db.get_or_create_collection(
        name=f"resumes_{org_id}",
        metadata={"hnsw:space": "cosine"}
    )

    embedder = get_embeddings()
    texts, metadatas, ids = [], [], []

    for i, chunk in enumerate(chunks):
        texts.append(chunk.page_content)
        metadatas.append({
            "source_file": filename,
            "candidate_name": profile.get("full_name", "Unknown"),
            "email": profile.get("email", ""),
            "phone": profile.get("phone", ""),
            "current_role": profile.get("current_role", ""),
            "chunk_id": i,
            "profile_json": json.dumps(profile)
        })
        ids.append(f"{filename}_{i}_{int(time.time())}")

    embeddings = embedder.embed_documents(texts)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    # Also store profile in session state
    if "candidate_profiles" not in st.session_state:
        st.session_state.candidate_profiles = {}
    st.session_state.candidate_profiles[filename] = profile

    return profile


# ─────────────────────────────────────────
# TALENT DNA MATCHING ENGINE
# ─────────────────────────────────────────

def match_candidates_to_jd(jd_text: str, org_id: str, top_k: int = 10) -> list:
    """
    THE CORE ENGINE:
    1. Embed the JD
    2. Semantic search across all resumes
    3. Group results by candidate
    4. Score each candidate with LLM
    5. Return ranked candidates with full analysis
    """
    db = get_vector_db()

    try:
        collection = db.get_collection(f"resumes_{org_id}")
    except:
        return []

    embedder = get_embeddings()
    jd_embedding = embedder.embed_query(jd_text)

    # Search top matching chunks
    results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=min(50, collection.count())
    )

    if not results or not results['documents'][0]:
        return []

    # Group by candidate
    candidate_chunks = {}
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        fname = meta['source_file']
        if fname not in candidate_chunks:
            candidate_chunks[fname] = {
                "chunks": [],
                "meta": meta,
                "profile": json.loads(
                    meta.get("profile_json", "{}")
                )
            }
        candidate_chunks[fname]["chunks"].append(doc)

    # Score each candidate with LLM
    llm = get_llm()
    scored_candidates = []

    for fname, data in candidate_chunks.items():
        profile = data["profile"]
        resume_summary = " ".join(data["chunks"][:5])

        scoring_prompt = f"""You are an elite HR analyst.
Score this candidate against the Job Description.
Return ONLY a valid JSON object.

JOB DESCRIPTION:
{jd_text[:1500]}

CANDIDATE PROFILE:
Name: {profile.get('full_name', 'Unknown')}
Role: {profile.get('current_role', 'Unknown')}
Experience: {profile.get('total_experience_years', '?')} years
Skills: {', '.join(profile.get('skills', [])[:15])}
Education: {profile.get('education', 'Unknown')}
Summary: {resume_summary[:800]}

SCORING JSON:
{{
  "overall_score": <0-100 integer>,
  "skill_match_score": <0-100>,
  "experience_score": <0-100>,
  "education_score": <0-100>,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "bonus_skills": ["extra skill not in JD"],
  "hire_recommendation": "STRONG YES / YES / MAYBE / NO",
  "one_liner": "One sentence honest assessment",
  "red_flags": ["flag1"] or [],
  "strengths": ["strength1", "strength2"],
  "interview_questions": ["q1", "q2", "q3"]
}}"""

        try:
            resp = llm.invoke(scoring_prompt)
            content = resp.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            scoring = json.loads(content.strip())
        except:
            scoring = {
                "overall_score": 40,
                "skill_match_score": 40,
                "experience_score": 40,
                "education_score": 40,
                "matching_skills": [],
                "missing_skills": [],
                "bonus_skills": [],
                "hire_recommendation": "MAYBE",
                "one_liner": "Could not fully analyze",
                "red_flags": [],
                "strengths": [],
                "interview_questions": []
            }

        scored_candidates.append({
            "profile": profile,
            "scoring": scoring,
            "filename": fname
        })

    # Sort by overall score
    scored_candidates.sort(
        key=lambda x: x["scoring"].get("overall_score", 0),
        reverse=True
    )

    return scored_candidates[:top_k]


# ─────────────────────────────────────────
# UTILITY: SCORE COLOR
# ─────────────────────────────────────────

def get_score_class(score: int) -> str:
    if score >= 80: return "score-excellent"
    elif score >= 65: return "score-good"
    elif score >= 45: return "score-average"
    else: return "score-poor"

def get_score_color(score: int) -> str:
    if score >= 80: return "#00ff88"
    elif score >= 65: return "#ffd700"
    elif score >= 45: return "#ff8c00"
    else: return "#ff4757"

def get_rank_badge(rank: int) -> str:
    if rank == 1: return f'<span class="rank-1">🥇 #1 TOP PICK</span>'
    elif rank == 2: return f'<span class="rank-2">🥈 #2 RUNNER UP</span>'
    elif rank == 3: return f'<span class="rank-3">🥉 #3 SHORTLIST</span>'
    else: return f'<span class="rank-other">#{rank}</span>'

def get_rec_emoji(rec: str) -> str:
    mapping = {
        "STRONG YES": "🟢",
        "YES": "🔵",
        "MAYBE": "🟡",
        "NO": "🔴"
    }
    for k, v in mapping.items():
        if k in rec.upper():
            return v
    return "⚪"


# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────

if "candidate_profiles" not in st.session_state:
    st.session_state.candidate_profiles = {}
if "match_results" not in st.session_state:
    st.session_state.match_results = []
if "active_jd" not in st.session_state:
    st.session_state.active_jd = ""
if "shortlist" not in st.session_state:
    st.session_state.shortlist = []


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:

    # Logo
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.5em;">🎯</div>
        <div style="font-size:1.4em; font-weight:700;
             background: linear-gradient(135deg, #667eea, #f64f59);
             -webkit-background-clip: text;
             -webkit-text-fill-color: transparent;">
             HireIQ
        </div>
        <div style="font-size:11px; color:#666;
             letter-spacing:2px; margin-top:2px;">
             TALENT INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Company Setup
    st.markdown("**🏢 Company**")
    company = st.text_input(
        "Company Name",
        value="My Company",
        label_visibility="collapsed",
        placeholder="Enter company name..."
    )
    org_id = company.lower().replace(" ", "_").replace("-", "_")

    st.divider()

    # Resume Upload
    st.markdown("**📋 Upload Resumes**")
    st.caption("PDF · DOCX · TXT supported")

    uploaded_resumes = st.file_uploader(
        "Drop resumes",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'doc'],
        label_visibility="collapsed"
    )

    if uploaded_resumes:
        if st.button(
            "⚡ Scan & Index All Resumes",
            type="primary",
            use_container_width=True
        ):
            progress = st.progress(0)
            status = st.empty()
            indexed = 0

            for i, file in enumerate(uploaded_resumes):
                status.markdown(
                    f"🔍 Scanning **{file.name}**..."
                )
                docs = load_resume(file)
                if docs:
                    profile = index_resume(
                        docs, file.name, org_id
                    )
                    indexed += 1
                    st.success(
                        f"✅ **{profile.get('full_name', file.name)}**"
                    )
                progress.progress((i + 1) / len(uploaded_resumes))

            status.empty()
            st.balloons()
            st.success(f"🎉 {indexed} resumes indexed!")

    st.divider()

    # Talent Pool Stats
    st.markdown("**📊 Talent Pool**")

    try:
        db = get_vector_db()
        col = db.get_collection(f"resumes_{org_id}")
        total_chunks = col.count()
        # Estimate candidates
        candidate_count = len(
            set([
                m['source_file']
                for m in col.get()['metadatas']
            ])
        ) if total_chunks > 0 else 0
    except:
        total_chunks = 0
        candidate_count = 0

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{candidate_count}</div>
            <div class="stats-label">Candidates</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">
                {len(st.session_state.shortlist)}
            </div>
            <div class="stats-label">Shortlisted</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Clear DB
    if st.button(
        "🗑️ Clear Talent Pool",
        use_container_width=True
    ):
        try:
            db = get_vector_db()
            db.delete_collection(f"resumes_{org_id}")
            st.session_state.candidate_profiles = {}
            st.session_state.match_results = []
            st.session_state.shortlist = []
            st.success("✅ Pool cleared!")
            st.rerun()
        except:
            st.info("Nothing to clear")


# ─────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────

# Hero Banner
st.markdown(f"""
<div class="hero-banner">
    <h1>🎯 HireIQ</h1>
    <div class="tagline">
        AI-Powered Talent Intelligence · Find Your Perfect Hire in Seconds
    </div>
    <div class="badge">⚡ Powered by Local AI · 100% Private · Zero Cloud</div>
</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  JD Matcher",
    "👥  Talent Pool",
    "📋  Shortlist",
    "💬  AI Recruiter Chat"
])


# ═══════════════════════════════════════════
# TAB 1: JD MATCHER
# ═══════════════════════════════════════════

with tab1:

    st.markdown("""
    <div class="section-header">
        <span>🎯</span>
        <h3>Job Description Intelligence Matcher</h3>
    </div>
    """, unsafe_allow_html=True)

    # JD Input area
    jd_col, config_col = st.columns([3, 1])

    with jd_col:
        st.markdown("""
        <div class="jd-input-box">
        """, unsafe_allow_html=True)

        jd_text = st.text_area(
            "📝 Paste your Job Description here",
            value=st.session_state.active_jd,
            height=280,
            placeholder="""Example:

We are looking for a Senior Python Developer with:
- 5+ years Python experience
- FastAPI / Django experience
- Knowledge of AWS / GCP
- Strong SQL and NoSQL database skills
- Experience with Docker and Kubernetes
- Good communication skills

Bonus: ML/AI experience, Team leadership""",
            label_visibility="visible"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with config_col:
        st.markdown("**⚙️ Match Settings**")
        top_k = st.slider(
            "Top Candidates", 1, 20, 5
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Quick JD templates
        st.markdown("**📄 Quick Templates**")
        templates = {
            "Python Dev": "Senior Python developer, 5+ years, FastAPI Django AWS Docker Kubernetes REST APIs PostgreSQL",
            "Data Scientist": "Data Scientist, 3+ years, Python R TensorFlow PyTorch ML models statistics SQL Tableau",
            "Product Manager": "Product Manager 4+ years, roadmap stakeholder management agile scrum user research metrics",
            "DevOps Eng": "DevOps Engineer, CI/CD Jenkins GitHub Actions Terraform AWS Kubernetes Docker monitoring",
            "UI/UX Designer": "UI/UX Designer, Figma Adobe XD user research wireframing prototyping design systems"
        }
        for tname, ttext in templates.items():
            if st.button(
                tname,
                key=f"tpl_{tname}",
                use_container_width=True
            ):
                st.session_state.active_jd = ttext
                st.rerun()

    # MATCH BUTTON
    st.markdown("<br>", unsafe_allow_html=True)

    match_col, _, _ = st.columns([2, 1, 1])
    with match_col:
        run_match = st.button(
            "🚀 Run Talent DNA Match",
            type="primary",
            use_container_width=True
        )

    if run_match:
        if not jd_text.strip():
            st.warning("⚠️ Please paste a Job Description first!")
        elif candidate_count == 0:
            st.warning("⚠️ Upload resumes first (sidebar)!")
        else:
            st.session_state.active_jd = jd_text
            with st.status(
                "🧬 Running Talent DNA Matching...",
                expanded=True
            ) as status:
                st.write("🔍 Embedding JD semantics...")
                time.sleep(0.3)
                st.write("🧠 Scanning talent pool...")
                results = match_candidates_to_jd(
                    jd_text, org_id, top_k
                )
                st.write(
                    f"📊 Scoring {len(results)} candidates..."
                )
                st.session_state.match_results = results
                status.update(
                    label="✅ Match Complete!",
                    state="complete"
                )

    # ── RESULTS ──────────────────────────
    if st.session_state.match_results:
        results = st.session_state.match_results

        st.markdown(f"""
        <div class="section-header">
            <span>🏆</span>
            <h3>Top {len(results)} Matched Candidates</h3>
        </div>
        """, unsafe_allow_html=True)

        # Summary bar
        avg_score = sum(
            r["scoring"].get("overall_score", 0)
            for r in results
        ) / len(results)
        strong_yes = sum(
            1 for r in results
            if "STRONG YES" in r["scoring"].get(
                "hire_recommendation", ""
            ).upper()
        )

        s1, s2, s3, s4 = st.columns(4)
        metrics = [
            (s1, len(results), "Candidates Analyzed"),
            (s2, f"{avg_score:.0f}%", "Avg Match Score"),
            (s3, strong_yes, "Strong Hires"),
            (s4, results[0]["scoring"].get(
                "overall_score", 0
            ), "Top Score")
        ]
        for col, val, label in metrics:
            with col:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{val}</div>
                    <div class="stats-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Candidate Cards
        for rank, candidate in enumerate(results, 1):
            profile = candidate["profile"]
            scoring = candidate["scoring"]
            score = scoring.get("overall_score", 0)
            name = profile.get("full_name", "Unknown")
            rec = scoring.get(
                "hire_recommendation", "MAYBE"
            )

            with st.expander(
                f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉' if rank==3 else '👤'}"
                f"  {name}  ·  {score}% Match  ·  "
                f"{get_rec_emoji(rec)} {rec}",
                expanded=(rank <= 2)
            ):
                left, right = st.columns([3, 2])

                with left:
                    # Rank + Name
                    st.markdown(
                        get_rank_badge(rank),
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"### {name}",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"*{profile.get('current_role','Unknown')} · "
                        f"{profile.get('total_experience_years','?')} yrs exp · "
                        f"{profile.get('location','Unknown')}*"
                    )

                    # Summary
                    st.markdown(
                        f"> {scoring.get('one_liner','')}"
                    )

                    # Contact Card
                    email = profile.get('email','Not found')
                    phone = profile.get('phone','Not found')
                    linkedin = profile.get('linkedin','Not found')
                    github = profile.get('github','Not found')

                    st.markdown(f"""
                    <div class="contact-card">
                        <div class="contact-item">
                            📧 <strong>{email}</strong>
                        </div>
                        <div class="contact-item">
                            📱 <strong>{phone}</strong>
                        </div>
                        {'<div class="contact-item">🔗 ' + linkedin + '</div>' if linkedin != 'Not found' else ''}
                        {'<div class="contact-item">💻 ' + github + '</div>' if github != 'Not found' else ''}
                    </div>
                    """, unsafe_allow_html=True)

                    # Education
                    st.markdown(
                        f"🎓 `{profile.get('education','Unknown')}`"
                    )

                    # Skills Match
                    st.markdown("**🧬 Skill DNA Analysis**")

                    matching = scoring.get(
                        "matching_skills", []
                    )
                    missing = scoring.get(
                        "missing_skills", []
                    )
                    bonus = scoring.get(
                        "bonus_skills", []
                    )

                    if matching:
                        st.markdown("✅ **Matched:**")
                        badges = " ".join([
                            f'<span class="talent-badge badge-match">{s}</span>'
                            for s in matching[:8]
                        ])
                        st.markdown(
                            badges,
                            unsafe_allow_html=True
                        )

                    if missing:
                        st.markdown("❌ **Missing:**")
                        badges = " ".join([
                            f'<span class="talent-badge badge-gap">{s}</span>'
                            for s in missing[:6]
                        ])
                        st.markdown(
                            badges,
                            unsafe_allow_html=True
                        )

                    if bonus:
                        st.markdown("⭐ **Bonus Skills:**")
                        badges = " ".join([
                            f'<span class="talent-badge badge-neutral">{s}</span>'
                            for s in bonus[:5]
                        ])
                        st.markdown(
                            badges,
                            unsafe_allow_html=True
                        )

                with right:
                    # Score Breakdown
                    st.markdown("**📊 Match Scores**")

                    score_items = [
                        ("Overall", score),
                        ("Skills", scoring.get(
                            "skill_match_score", 0
                        )),
                        ("Experience", scoring.get(
                            "experience_score", 0
                        )),
                        ("Education", scoring.get(
                            "education_score", 0
                        )),
                    ]

                    for label, s in score_items:
                        color = get_score_color(s)
                        st.markdown(
                            f"<small style='color:#888'>{label}</small>"
                            f"<span style='float:right;color:{color};font-weight:700'>{s}%</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown(f"""
                        <div class="match-bar-wrapper">
                            <div class="match-bar-fill"
                                 style="width:{s}%;background:linear-gradient(90deg,{color}88,{color})">
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Strengths
                    strengths = scoring.get("strengths", [])
                    if strengths:
                        st.markdown("**💪 Key Strengths**")
                        for s in strengths[:3]:
                            st.markdown(f"• {s}")

                    # Red Flags
                    flags = scoring.get("red_flags", [])
                    if flags:
                        st.markdown(
                            "**⚠️ Red Flags**",
                            help="Items to probe in interview"
                        )
                        for f in flags[:3]:
                            st.markdown(
                                f"⚠️ <small>{f}</small>",
                                unsafe_allow_html=True
                            )

                    # Interview Qs
                    iqs = scoring.get(
                        "interview_questions", []
                    )
                    if iqs:
                        st.markdown("**❓ Interview Questions**")
                        for q in iqs[:3]:
                            st.markdown(
                                f"<small>→ {q}</small>",
                                unsafe_allow_html=True
                            )

                    # Actions
                    st.markdown("<br>", unsafe_allow_html=True)
                    a1, a2 = st.columns(2)
                    with a1:
                        if st.button(
                            "⭐ Shortlist",
                            key=f"sl_{rank}_{name}",
                            use_container_width=True
                        ):
                            if candidate not in st.session_state.shortlist:
                                st.session_state.shortlist.append(candidate)
                                st.success("Added!")
                    with a2:
                        if st.button(
                            "📧 Draft Email",
                            key=f"em_{rank}_{name}",
                            use_container_width=True
                        ):
                            st.session_state[
                                f"show_email_{name}"
                            ] = True

                    # Email Drafter
                    if st.session_state.get(
                        f"show_email_{name}"
                    ):
                        llm = get_llm()
                        with st.spinner("✍️ Drafting..."):
                            email_prompt = f"""Write a professional interview invitation email.
Recruiter from: {company}
Candidate: {name}
Role applied: based on JD: {jd_text[:300]}
Tone: warm, professional, exciting
Keep it under 150 words."""
                            email_resp = llm.invoke(
                                email_prompt
                            )
                            st.text_area(
                                "📧 Interview Invitation",
                                value=email_resp.content,
                                height=200,
                                key=f"email_text_{name}"
                            )


# ═══════════════════════════════════════════
# TAB 2: TALENT POOL
# ═══════════════════════════════════════════

with tab2:

    st.markdown("""
    <div class="section-header">
        <span>👥</span>
        <h3>Full Talent Pool Directory</h3>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.candidate_profiles:
        st.info(
            "📭 No candidates indexed yet.\n\n"
            "Upload resumes in the sidebar to build your talent pool!"
        )
    else:
        profiles = st.session_state.candidate_profiles

        # Search + Filter Bar
        f1, f2 = st.columns([3, 1])
        with f1:
            search = st.text_input(
                "🔎 Search by name, role, skill...",
                placeholder="e.g. Python, React, Manager..."
            )
        with f2:
            sort_by = st.selectbox(
                "Sort by",
                ["Name", "Experience", "Role"]
            )

        # Filter profiles
        filtered = {}
        for fname, prof in profiles.items():
            if search:
                haystack = (
                    str(prof.get("full_name", "")) +
                    str(prof.get("current_role", "")) +
                    str(prof.get("skills", [])) +
                    str(prof.get("education", ""))
                ).lower()
                if search.lower() not in haystack:
                    continue
            filtered[fname] = prof

        st.markdown(
            f"<small style='color:#666'>Showing "
            f"{len(filtered)} of {len(profiles)} candidates</small>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Display Grid
        for fname, prof in filtered.items():
            with st.container():
                st.markdown(
                    '<div class="dna-card">',
                    unsafe_allow_html=True
                )
                c1, c2, c3, c4 = st.columns([3, 2, 2, 1])

                with c1:
                    st.markdown(
                        f"**{prof.get('full_name', 'Unknown')}**"
                    )
                    st.markdown(
                        f"<small style='color:#888'>"
                        f"{prof.get('current_role','Unknown')}"
                        f"</small>",
                        unsafe_allow_html=True
                    )
                    skills_preview = prof.get("skills", [])[:4]
                    badges = " ".join([
                        f'<span class="talent-badge badge-skill">{s}</span>'
                        for s in skills_preview
                    ])
                    st.markdown(
                        badges, unsafe_allow_html=True
                    )

                with c2:
                    st.markdown(
                        f"📧 `{prof.get('email','?')}`"
                    )
                    st.markdown(
                        f"📱 `{prof.get('phone','?')}`"
                    )
                    st.markdown(
                        f"📍 {prof.get('location','?')}"
                    )

                with c3:
                    st.markdown(
                        f"🎓 {prof.get('education','?')}"
                    )
                    st.markdown(
                        f"⏱️ **{prof.get('total_experience_years','?')}** years exp"
                    )
                    st.markdown(
                        f"🏢 {', '.join(prof.get('companies',[])[:2])}"
                    )

                with c4:
                    st.markdown(
                        f"*Indexed:*\n{prof.get('indexed_at','')}"
                    )

                st.markdown(
                    "</div>", unsafe_allow_html=True
                )
                st.markdown("")


# ═══════════════════════════════════════════
# TAB 3: SHORTLIST
# ═══════════════════════════════════════════

with tab3:

    st.markdown("""
    <div class="section-header">
        <span>📋</span>
        <h3>My Shortlist</h3>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.shortlist:
        st.info(
            "📭 Your shortlist is empty.\n\n"
            "Run a JD Match and click ⭐ Shortlist on candidates!"
        )
    else:
        st.markdown(
            f"**{len(st.session_state.shortlist)} "
            f"candidates shortlisted**"
        )

        # Export button
        if st.button("📥 Export Shortlist as CSV"):
            rows = []
            for c in st.session_state.shortlist:
                p = c["profile"]
                s = c["scoring"]
                rows.append({
                    "Name": p.get("full_name",""),
                    "Email": p.get("email",""),
                    "Phone": p.get("phone",""),
                    "LinkedIn": p.get("linkedin",""),
                    "Role": p.get("current_role",""),
                    "Experience": p.get("total_experience_years",""),
                    "Match Score": s.get("overall_score",""),
                    "Recommendation": s.get("hire_recommendation",""),
                    "Key Strength": ", ".join(
                        s.get("strengths",[])[:2]
                    )
                })
            df = pd.DataFrame(rows)
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False),
                "shortlist.csv",
                "text/csv"
            )
            st.dataframe(df, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for i, candidate in enumerate(
            st.session_state.shortlist
        ):
            p = candidate["profile"]
            s = candidate["scoring"]
            score = s.get("overall_score", 0)

            col_main, col_remove = st.columns([9, 1])

            with col_main:
                with st.expander(
                    f"⭐ {p.get('full_name','?')} · "
                    f"{score}% · {s.get('hire_recommendation','')}"
                ):
                    r1, r2 = st.columns(2)
                    with r1:
                        st.markdown("**Contact Info**")
                        st.markdown(
                            f"📧 {p.get('email','?')}"
                        )
                        st.markdown(
                            f"📱 {p.get('phone','?')}"
                        )
                        st.markdown(
                            f"🔗 {p.get('linkedin','N/A')}"
                        )
                        st.markdown(
                            f"💻 {p.get('github','N/A')}"
                        )
                    with r2:
                        st.markdown("**Assessment**")
                        st.markdown(
                            f"🎯 Score: **{score}%**"
                        )
                        st.markdown(
                            f"💡 {s.get('one_liner','')}"
                        )
                        st.markdown(
                            f"✅ Strengths: "
                            f"{', '.join(s.get('strengths',[])[:2])}"
                        )

            with col_remove:
                if st.button(
                    "✕",
                    key=f"rm_{i}",
                    help="Remove from shortlist"
                ):
                    st.session_state.shortlist.pop(i)
                    st.rerun()


# ═══════════════════════════════════════════
# TAB 4: AI RECRUITER CHAT
# ═══════════════════════════════════════════

with tab4:

    st.markdown("""
    <div class="section-header">
        <span>💬</span>
        <h3>AI Recruiter — Ask Anything About Your Talent Pool</h3>
    </div>
    """, unsafe_allow_html=True)

    # Init chat
    if "recruiter_chat" not in st.session_state:
        st.session_state.recruiter_chat = [{
            "role": "assistant",
            "content": (
                "👋 Hi! I'm your **AI Recruiter**.\n\n"
                "I can answer questions about your talent pool, "
                "compare candidates, suggest interview strategies, "
                "and help you make hiring decisions.\n\n"
                "**Try asking:**\n"
                "- *Who has the strongest Python skills?*\n"
                "- *Compare the top 2 candidates*\n"
                "- *Which candidates have AWS experience?*\n"
                "- *Suggest interview questions for [name]*\n"
                "- *Who is the best culture fit for a startup?*"
            )
        }]

    # Display messages
    for msg in st.session_state.recruiter_chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if user_q := st.chat_input(
        "Ask about your candidates..."
    ):
        st.session_state.recruiter_chat.append({
            "role": "user",
            "content": user_q
        })
        with st.chat_message("user"):
            st.write(user_q)

        # Build context from profiles + match results
        pool_context = ""
        if st.session_state.candidate_profiles:
            for fname, p in list(
                st.session_state.candidate_profiles.items()
            )[:10]:
                pool_context += (
                    f"\n---\nName: {p.get('full_name')}\n"
                    f"Role: {p.get('current_role')}\n"
                    f"Exp: {p.get('total_experience_years')}yrs\n"
                    f"Skills: {', '.join(p.get('skills',[])[:8])}\n"
                    f"Email: {p.get('email')}\n"
                    f"Education: {p.get('education')}\n"
                )

        match_context = ""
        if st.session_state.match_results:
            match_context = "RECENT MATCH RESULTS:\n"
            for r in st.session_state.match_results[:5]:
                p = r["profile"]
                s = r["scoring"]
                match_context += (
                    f"{p.get('full_name')}: "
                    f"{s.get('overall_score')}% match, "
                    f"{s.get('hire_recommendation')}\n"
                )

        llm = get_llm()
        chat_prompt = f"""You are an elite AI Recruiter and Talent Advisor.
You have deep knowledge of hiring, interviews, and talent assessment.

TALENT POOL DATA:
{pool_context if pool_context else "No candidates indexed yet."}

{match_context}

ACTIVE JD:
{st.session_state.active_jd[:500] if st.session_state.active_jd else "No JD set."}

RECRUITER QUESTION:
{user_q}

Give a specific, helpful, actionable answer.
Reference actual candidates by name when relevant.
Be concise but insightful."""

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = llm.invoke(chat_prompt)
                st.write(response.content)

        st.session_state.recruiter_chat.append({
            "role": "assistant",
            "content": response.content
        })