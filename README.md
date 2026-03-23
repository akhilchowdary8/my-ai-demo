verison new
SCREEN FLOW (Like a real Enterprise HR Tool)
─────────────────────────────────────────────────────────────

SCREEN 1: COMMAND CENTER (Dashboard)
──────────────────────────────────────
┌─────────────────────────────────────────────────────────┐
│  🎯 HireIQ Pro          [Company: Acme Corp]  [Settings]│
│  ─────────────────────────────────────────────────────  │
│  📊 142 Resumes   🎯 3 Active JDs   ✅ 28 Shortlisted  │
│  ─────────────────────────────────────────────────────  │
│                                                          │
│  STEP 1          STEP 2          STEP 3        STEP 4   │
│  📁 Upload   →  🎯 Match JD  →  👥 Review  →  📧 Send  │
│  Resumes         Paste JD        Shortlist      Emails  │
└─────────────────────────────────────────────────────────┘


SCREEN 2: BULK RESUME UPLOAD
──────────────────────────────
┌─────────────────────────────────────────────────────────┐
│  DROP ZONE (Drag & Drop 100s of files at once)          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  📁 Drop ALL resumes here  (PDF/DOCX/TXT)        │  │
│  │  or click to browse — select 100s at once        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  LIVE INDEXING DASHBOARD:                               │
│  ████████████░░░░  67/100 processed  |  ETA: 45s        │
│                                                          │
│  ✅ John Smith      — Indexed (8 sections found)        │
│  ✅ Sarah Connor    — Indexed (6 sections found)        │
│  ⏳ Mike Johnson   — Processing...                      │
│  ⏸️  45 more in queue                                   │
└─────────────────────────────────────────────────────────┘


SCREEN 3: JD MATCHER + THRESHOLD
──────────────────────────────────
┌─────────────────────────────────────────────────────────┐
│  PASTE JD:                                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ We are looking for...                             │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  HR THRESHOLD:  60──────●──────90  [75]                 │
│                                                          │
│  [🚀 RUN TALENT DNA MATCH]                             │
└─────────────────────────────────────────────────────────┘


SCREEN 4: RESULTS LEADERBOARD
───────────────────────────────
┌─────────────────────────────────────────────────────────┐
│  🏆 QUALIFIED (75+ score) — 12 candidates               │
│  ─────────────────────────────────────────────────────  │
│  #1  John Smith    ██████████  94%  ✅ STRONG YES       │
│  #2  Sarah Lee     █████████   88%  ✅ YES              │
│  #3  Mike Brown    ████████    81%  ✅ YES              │
│                                                          │
│  ⚠️ BORDERLINE (60-74) — 8 candidates                  │
│  ─────────────────────────────────────────────────────  │
│  #4  Amy Chen      ███████     72%  🟡 MAYBE           │
│                                                          │
│  ❌ NOT QUALIFIED (<60) — 22 candidates                 │
│  ─────────────────────────────────────────────────────  │
│  #12 Bob Jones     ████        48%  ❌ NO              │
└─────────────────────────────────────────────────────────┘


SCREEN 5: EMAIL CENTER
────────────────────────
┌─────────────────────────────────────────────────────────┐
│  SELECT MODE:  ○ Individual  ● Bulk Shortlist           │
│  ─────────────────────────────────────────────────────  │
│  EMAIL TEMPLATE: [Interview Invite ▼]                   │
│                                                          │
│  TO:  [Auto-filled from shortlist]                      │
│  BCC: [hr@company.com]                                  │
│                                                          │
│  DRAFT (AI Generated, HR can edit):                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Dear {candidate_name},                            │  │
│  │ We reviewed your profile for {role}...            │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  [👁️ Preview]  [✏️ Edit]  [✅ Approve & Send All]      │
└─────────────────────────────────────────────────────────┘



COMPONENT          MODEL              SPEED    ACCURACY
─────────────────────────────────────────────────────────
Resume Parsing     phi3:mini          ⚡⚡⚡    ★★★★
Section Detection  phi3:mini          ⚡⚡⚡    ★★★★
JD Analysis        llama3.2           ⚡⚡      ★★★★★
Scoring            llama3.2           ⚡⚡      ★★★★★
Email Drafting     llama3.2           ⚡⚡      ★★★★★
Embeddings         nomic-embed-text   ⚡⚡⚡    ★★★★★

WHY phi3:mini for parsing:
→ Microsoft's most efficient small model
→ 3x faster than llama3.2 for extraction tasks
→ Same accuracy for structured JSON output
→ Perfect for processing 100s of resumes fast

WHY llama3.2 for matching/email:
→ Better reasoning for complex comparison
→ Better language quality for email drafts
→ Only called ONCE per JD match, not per resume
