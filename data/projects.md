# Projects — Francesco Colasurdo

## Mental Health Conversational AI (NTT Data, Naples)
Role: Data Science Intern  
Period: Mar 2025 – Jun 2025  
Location: Naples, Italy

### Problem
Mental well-being conversations are sensitive, high-volume, and highly contextual. The company needed a scalable conversational AI capable of interacting with users, detecting emotional state, and producing actionable insights for analysis — in real time.

### What I built
- I designed and deployed a mental health chatbot using Dialogflow CX integrated with Telegram.
- I connected the chatbot to Google BigQuery and collected >1,000 conversations.
- I implemented clustering and sentiment analysis to identify behavioral patterns in users.

### Technical Stack
- Dialogflow CX (intent detection, conversation flow)
- Telegram Bot API (interaction layer)
- Gemini API (LLM-based sentiment and emotional tone detection)
- HDBSCAN + UMAP (unsupervised clustering of conversation embeddings)
- R (regression analysis, exploratory data analysis, hypothesis testing)
- Power BI (interactive dashboards for insights and storytelling)
- BigQuery (data storage, querying, live monitoring)

### Impact
- >85% completion rate on conversations.
- 7 recurring behavioral clusters discovered across users.
- Built real-time dashboards that visualized emotional trends and engagement.
- This project formed the core of my Bachelor's thesis.

---

## Market Intelligence Automation (Italian–South African Chamber of Trade and Industries)
Role: Data Analytics & Market Research Intern  
Period: Jan 2025 – Mar 2025  
Location: Johannesburg, South Africa

### Problem
Business partners and export stakeholders needed fast, data-backed answers about high-growth sectors (energy, steel, agri-food). The manual process to gather this info was slow (several minutes per query) and not standardized.

### What I built
- I produced 4 full market analyses identifying >20% annual growth opportunities in strategic sectors.
- I built an internal market-intelligence database and automated data collection and reporting flows.
- I connected Excel data sources and Power BI dashboards to deliver near-instant answers.

### Technical Stack
- Power BI (dashboards, KPIs, trend visualization)
- Excel automation (ETL-like internal workflow)
- Data cleaning and consolidation for repeatable intelligence

### Impact
- Reduced time-to-answer from ~3 minutes to <15 seconds.
- Improved decision-making speed for strategic partnership and export opportunities between Italian and South African firms.
- Helped leadership maintain updated sector views without manual digging.

---

## Ask My CV — Personal RAG Assistant
Role: Creator / Architect  
Period: 2025 – ongoing  
Location: Eindhoven / Remote

### Problem
Recruiters and collaborators often ask similar questions: "What have you worked on?", "What are your skills?", "What motivates you?". Most personal websites are static and do not reflect the real depth of a profile.

### What I built
- I designed a Retrieval-Augmented Generation (RAG) backend in Python using FastAPI.
- I store my personal knowledge base as Markdown files (cv.md, skills.md, courses.md, projects.md, about_me.md).
- I generate embeddings, perform cosine-similarity retrieval, and build first-person answers ("I worked on...").
- I integrated a lightweight chat widget into my personal website (Hugo + JS) so visitors can ask questions.

### Technical Stack
- FastAPI (Python API /api/chat)
- sentence-transformers for embeddings
- cosine similarity search over vectors.json
- datapizza-ai for agent orchestration and tool calling
- custom prompt engineering ("speak in first person, don't hallucinate")
- front-end chat widget embedded in a Hugo static site

### Impact
- Live, interactive portfolio instead of static CV.
- Lets me explain projects, values, experiences, motivations in my own voice.
- Helps me stand out to recruiters as someone who builds, not just studies.

---

## Blockchain / DeFi Research & Personal Studies
Role: Independent Researcher / Enthusiast  
Period: 2021 – 2024

### Focus
- Studied decentralized finance (DeFi) from a technological, economic, and philosophical viewpoint.
- Explored incentive design, tokenomics, and game theory in decentralized systems.
- Investigated how Bitcoin and blockchain challenge traditional power structures in terms of privacy, self-custody, financial sovereignty, and censorship resistance.

### Why it matters
- This mindset influences how I approach AI ethics, data ownership, and digital autonomy.
- I am interested in aligning AI with individual freedom, transparency, and accountability – not just performance.

### Keywords
`DeFi`, `blockchain`, `Bitcoin philosophy`, `privacy`, `self-sovereignty`, `tokenomics`, `game theory`, `financial inclusion`, `decentralization`, `tech for freedom`.
