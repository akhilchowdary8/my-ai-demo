# app.py
# Enterprise AI Gateway - RAG Demo
# Run with: streamlit run app.py

import streamlit as st
import chromadb
import os
import time
import tempfile

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader
)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

st.set_page_config(
    page_title="OrgAI - Your Company Assistant",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────
# FREE LOCAL COMPONENTS
# ─────────────────────────────────────────

@st.cache_resource
def get_llm():
    """Free local LLM via Ollama"""
    return ChatOllama(
        model="llama3.2",
        temperature=0.1
    )

@st.cache_resource
def get_embeddings():
    """Free local embeddings via Ollama"""
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )

@st.cache_resource
def get_vector_db():
    """Free local vector database"""
    return chromadb.PersistentClient(
        path="./orgai_data"
    )

# ─────────────────────────────────────────
# DOCUMENT PROCESSING
# ─────────────────────────────────────────

def load_document(uploaded_file):
    """
    Load any file type into text
    Supports: PDF, CSV, TXT, DOCX, MD, PY, JSON
    """
    # Save file temporarily
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
        elif ext == 'csv':
            loader = CSVLoader(tmp_path)
        elif ext in ['doc', 'docx']:
            loader = Docx2txtLoader(tmp_path)
        elif ext in ['txt', 'md', 'py', 'js', 'json']:
            loader = TextLoader(tmp_path)
        else:
            st.error(f"❌ Unsupported: {ext}")
            return None

        docs = loader.load()
        os.unlink(tmp_path)
        return docs

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None


def process_and_store(docs, org_id, source_name):
    """
    Split documents into chunks
    Convert to embeddings
    Store in ChromaDB
    """
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # Get or create org collection
    db = get_vector_db()
    collection = db.get_or_create_collection(
        name=f"org_{org_id}",
        metadata={"hnsw:space": "cosine"}
    )

    embedder = get_embeddings()

    texts = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        texts.append(chunk.page_content)
        metadatas.append({
            "source": source_name,
            "org": org_id,
            "chunk_id": i
        })
        ids.append(
            f"{source_name}_{i}_{int(time.time())}"
        )

    # Create embeddings locally (free)
    embeddings = embedder.embed_documents(texts)

    # Store in ChromaDB (free)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return len(chunks)


def search_knowledge(query, org_id, n_results=5):
    """
    Search org knowledge base
    Returns most relevant chunks
    """
    db = get_vector_db()

    try:
        collection = db.get_collection(
            f"org_{org_id}"
        )
    except:
        return None

    embedder = get_embeddings()
    query_embedding = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results


def ask_question(question, org_id):
    """
    MAIN FUNCTION:
    1. Search org knowledge
    2. Build context from results
    3. Ask LLM to answer using ONLY org context
    4. Return answer + sources
    """

    # Step 1: Search
    results = search_knowledge(question, org_id)

    if not results or not results['documents'][0]:
        return (
            "⚠️ I don't have information about this "
            "in your organization's knowledge base.\n\n"
            "Please upload relevant documents first!",
            []
        )

    # Step 2: Build context
    context_parts = []
    sources = []

    for i, doc in enumerate(results['documents'][0]):
        source = results['metadatas'][0][i]['source']
        context_parts.append(
            f"[Source: {source}]\n{doc}"
        )
        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Ask LLM
    llm = get_llm()

    prompt = f"""You are an AI assistant for this 
specific organization.

STRICT RULES:
- Answer ONLY from the company context below
- If answer is not in context, say:
  "I don't have this in your company documents"
- Always be specific and cite details
- Never use outside knowledge
- Keep answers clear and concise

COMPANY KNOWLEDGE:
{context}

QUESTION: {question}

ANSWER:"""

    response = llm.invoke(prompt)

    return response.content, sources


# ─────────────────────────────────────────
# USER INTERFACE
# ─────────────────────────────────────────

# Styling
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(
            135deg, #1e3a5f, #2d6a9f
        );
        padding: 25px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
    .header-box h1 {
        margin: 0;
        font-size: 2em;
    }
    .header-box p {
        margin: 5px 0 0 0;
        opacity: 0.85;
    }
    .source-tag {
        background: #e8f4f8;
        border-left: 4px solid #2d6a9f;
        padding: 6px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 13px;
        color: #1e3a5f;
    }
    .stat-box {
        background: #f0f7ff;
        border: 1px solid #d0e8ff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stat-box h2 {
        color: #2d6a9f;
        font-size: 2.5em;
        margin: 0;
    }
    .stat-box p {
        color: #666;
        margin: 5px 0 0 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-box">
    <h1>🧠 OrgAI</h1>
    <p>Your Organization's Personal AI Assistant
     — Powered by Your Own Data</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────
with st.sidebar:

    # Org Setup
    st.header("🏢 Organization Setup")
    org_name = st.text_input(
        "Your Organization Name",
        value="My Company",
        placeholder="e.g. Alderno Financial"
    )
    org_id = org_name.lower().replace(
        " ", "_"
    ).replace("-", "_")

    st.divider()

    # File Upload
    st.header("📁 Upload Company Files")
    st.caption(
        "PDF, Word, CSV, TXT, Code files supported"
    )

    uploaded_files = st.file_uploader(
        "Drop files here",
        accept_multiple_files=True,
        type=[
            'pdf', 'txt', 'csv', 'docx',
            'md', 'py', 'js', 'json'
        ],
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button(
            "🚀 Process & Learn Files",
            type="primary",
            use_container_width=True
        ):
            total = 0
            bar = st.progress(0)
            status = st.empty()

            for i, file in enumerate(uploaded_files):
                status.text(
                    f"Learning: {file.name}..."
                )
                docs = load_document(file)

                if docs:
                    chunks = process_and_store(
                        docs, org_id, file.name
                    )
                    total += chunks
                    st.success(
                        f"✅ {file.name} "
                        f"({chunks} chunks)"
                    )

                bar.progress(
                    (i + 1) / len(uploaded_files)
                )

            status.empty()
            st.balloons()
            st.success(
                f"🎉 Done! {total} knowledge "
                f"chunks stored!"
            )

    st.divider()

    # URL Connector
    st.header("🌐 Connect a URL")
    st.caption("Learn from any webpage")

    url_input = st.text_input(
        "Paste URL here",
        placeholder="https://yourcompany.com/docs"
    )

    if st.button(
        "📥 Fetch & Learn",
        use_container_width=True
    ):
        if url_input:
            with st.spinner("Reading webpage..."):
                try:
                    from langchain_community\
                        .document_loaders\
                        import WebBaseLoader

                    loader = WebBaseLoader(url_input)
                    docs = loader.load()
                    chunks = process_and_store(
                        docs, org_id, url_input
                    )
                    st.success(
                        f"✅ Learned {chunks} "
                        f"chunks from URL!"
                    )
                except Exception as e:
                    st.error(f"❌ Failed: {e}")

    st.divider()

    # Knowledge Stats
    st.header("📊 Knowledge Base Status")

    try:
        db = get_vector_db()
        col = db.get_collection(f"org_{org_id}")
        count = col.count()

        st.markdown(f"""
        <div class="stat-box">
            <h2>{count}</h2>
            <p>Knowledge chunks stored<br>
            for {org_name}</p>
        </div>
        """, unsafe_allow_html=True)

        if count > 0:
            st.success("✅ AI ready to answer!")
        else:
            st.warning("⚠️ Upload files to start")

    except:
        st.markdown("""
        <div class="stat-box">
            <h2>0</h2>
            <p>No knowledge yet<br>
            Upload files above</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Example Questions
    st.header("💡 Example Questions")

    examples = [
        "What is our refund policy?",
        "Who handles HR issues?",
        "What is the approval process?",
        "What tools do we use?",
        "How do I contact the finance team?"
    ]

    for q in examples:
        if st.button(
            q,
            use_container_width=True,
            key=f"ex_{q}"
        ):
            st.session_state.example_q = q

    st.divider()

    # Clear Knowledge Base
    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True
    ):
        try:
            db = get_vector_db()
            db.delete_collection(f"org_{org_id}")
            st.success("✅ Knowledge base cleared!")
            st.rerun()
        except:
            st.info("Nothing to clear")


# ── MAIN CHAT AREA ────────────────────────

# Top bar
col1, col2 = st.columns([4, 1])
with col1:
    st.header(f"💬 Chat with {org_name}'s AI")
with col2:
    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            f"👋 Hello! I'm **{org_name}'s** "
            f"AI Assistant.\n\n"
            f"I answer questions using **only your "
            f"organization's data** — not the internet.\n\n"
            f"**Get started:**\n"
            f"1. 📁 Upload your company files (sidebar)\n"
            f"2. 💬 Ask me anything about your org\n\n"
            f"I'll only use YOUR documents to answer. "
            f"Nothing else."
        ),
        "sources": []
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander(
                f"📚 {len(msg['sources'])} "
                f"source(s) used"
            ):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-tag">'
                        f'📄 {src}</div>',
                        unsafe_allow_html=True
                    )

# Handle example question clicks
if hasattr(st.session_state, 'example_q'):
    query = st.session_state.example_q
    del st.session_state.example_q

    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "sources": []
    })

    with st.spinner("🔍 Searching org knowledge..."):
        answer, sources = ask_question(
            query, org_id
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.rerun()

# Chat input box
if user_input := st.chat_input(
    f"Ask anything about {org_name}..."
):
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "sources": []
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner(
            "🔍 Searching your org knowledge..."
        ):
            answer, sources = ask_question(
                user_input, org_id
            )

        st.write(answer)

        if sources:
            with st.expander(
                f"📚 {len(sources)} source(s) used"
            ):
                for src in sources:
                    st.markdown(
                        f'<div class="source-tag">'
                        f'📄 {src}</div>',
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })