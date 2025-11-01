# Thesis & Internship Project — Intelligent ChatBot for Psychological Well-being

## Overview
As part of my Bachelor's degree in **Statistics for Big Data** at the **University of Salerno (UNISA)**, I completed an internship and thesis project at **NTT Data Italia, Naples**, within the **Digital Linguistics & Human Behaviour Team**.  
The project focused on the **design, implementation, and statistical analysis** of a conversational AI system aimed at supporting **mental health and emotional self-awareness** through data-driven interaction.

## Objectives
- Design and deploy an **AI-powered conversational assistant** for psychological well-being monitoring.  
- Integrate advanced **Natural Language Processing (NLP)** with **statistical learning** and **data visualization**.  
- Collect, preprocess, and analyze **real user-chat data** to extract behavioral and emotional insights.  
- Build a **scalable, privacy-compliant data pipeline** for continuous improvement of the chatbot.

## System Architecture
### Core Components
1. **DialogFlow CX (Google Cloud)** — primary conversational flow engine.  
   - Designed multi-flow architecture (emotion flows: happiness, sadness, anxiety, boredom).  
   - Combined deterministic and generative routes for robust intent handling.  
   - Integrated custom Generators powered by **Google Gemini 1.5 Flash** for sentiment understanding.

2. **Telegram Bot API + Node.js Middleware**  
   - Developed a custom **API layer** to bridge Telegram and DialogFlow CX.  
   - Implemented a three-tier model: **User Interface (Telegram)** → **Middleware (Node.js)** → **Conversational Engine (DialogFlow CX)**.  
   - Managed user sessions, webhooks, and asynchronous message streaming.

3. **Google BigQuery (Data Warehouse)**  
   - Real-time storage of conversation logs in **JSON format**.  
   - Complex **SQL queries and CTEs** for flattening nested JSON data.  
   - Data anonymization and parameter extraction (e.g., sleep hours, stress level, productivity).

4. **Statistical & Machine Learning Layer (R Language)**  
   - Conducted advanced **Exploratory Data Analysis (EDA)** on structured data.  
   - Applied **Regression Analysis**, **DBSCAN**, and **HDBSCAN** clustering.  
   - Implemented **TF-IDF**, **sBERT embeddings**, and **UMAP** for textual feature representation and dimensionality reduction.  
   - Evaluated cluster validity with **Silhouette Score** and **Dunn Index**.

5. **Visualization Layer (Power BI)**  
   - Built interactive dashboards to display emotional trends, routine correlations, and ChatBot performance metrics.  
   - Enabled dynamic filtering by emotion, activity, and user behavior.

---

## Data Pipeline Summary
1. **Data Acquisition:** Conversations stored in BigQuery in real-time via DialogFlow logs.  
2. **Parsing & Wrangling:** Flattened nested JSON structures with SQL (`JSON_VALUE`, `REGEXP_EXTRACT`, `LAG`, `DATETIME_DIFF`).  
3. **Feature Engineering:** Extracted behavioral and emotional parameters per session (diet, sleep, hydration, productivity, anxiety level).  
4. **Text Preprocessing:** Tokenization, lemmatization, stopword removal, lowercasing, and normalization for uniformity.  
5. **Numerical Representation:**  
   - TF-IDF for lexical similarity.  
   - **sBERT (Sentence-BERT)** embeddings for semantic similarity.  
6. **Dimensionality Reduction:** UMAP projections for cluster visualization.  
7. **Clustering & Analysis:**  
   - DBSCAN → density-based clusters of similar messages.  
   - HDBSCAN → hierarchical and noise-tolerant clusters.  
8. **Evaluation & Insights:** Computed silhouette width and Dunn index to assess cluster cohesion and separation.

---

## Key Results & Insights
- **Identified behavioral patterns** correlating sleep duration, hydration, and self-reported productivity.  
- Clustered emotional states into distinct profiles (e.g., anxiety-driven users vs. happiness clusters).  
- Detected **linguistic signatures** tied to emotional states using TF-IDF and sBERT representations.  
- The ChatBot demonstrated **adaptive sentiment recognition** through real-time LLM (Gemini) feedback loops.  
- Achieved a **mean silhouette score of 0.31 (TF-IDF)** and **0.27 (sBERT)** with well-separated semantic clusters.  
- Generated a Power BI dashboard enabling **business and behavioral analytics** on user emotional states.

---

## Technical Stack
| Category | Technologies & Tools |
|-----------|----------------------|
| Conversational AI | **Google DialogFlow CX**, **Google Gemini LLMs**, **Prompt Engineering (CO-STAR, Zero-Shot, Few-Shot)** |
| Data Storage | **Google BigQuery**, SQL (CTE, JSON parsing, query optimization) |
| Backend Integration | **Node.js**, **Telegram Bot API**, **Webhooks**, **Local Server Bridge** |
| Data Science & ML | **R Language**, **TF-IDF**, **sBERT**, **UMAP**, **DBSCAN**, **HDBSCAN**, **Clustering Validation Metrics** |
| Visualization | **Microsoft Power BI**, interactive reports, trend dashboards |
| Cloud & Dev Tools | **Google Cloud Platform**, **VSCode**, **GitHub**, **JSON**, **CSV**, **APIs** |

---

## Methodological Highlights
- **Prompt Engineering:** Designed controlled prompts for consistent sentiment classification (Gemini-based generator).  
- **Unsupervised Learning:** Used density-based clustering to detect recurring psychological topics in open-ended responses.  
- **NLP Pipeline:** Combined symbolic NLP (TF-IDF) and transformer-based embeddings (sBERT).  
- **Dimensionality Reduction:** Applied UMAP to visualize text embeddings and improve clustering interpretability.  
- **Statistical Validation:** Used correlation matrices (Spearman) to link daily habits with emotional outcomes.

---

## Impact & Contributions
- Delivered a **fully functional conversational system** that collects and analyzes user emotional data in real time.  
- Designed a **scalable, cloud-based pipeline** for conversational data engineering and behavioral analysis.  
- Demonstrated how **AI and Data Science can enhance digital mental health monitoring**.  
- Provided empirical insights into the **relationship between habits, lifestyle, and emotional states**.  
- Enabled a reproducible workflow for future **AI-driven psychometric studies**.

---

## Competencies Developed
- Conversational AI engineering (DialogFlow CX, NLP design).  
- Data engineering with Google Cloud BigQuery (ETL, SQL on JSON).  
- Statistical modeling and clustering in R.  
- NLP pipeline development (TF-IDF, embeddings, transformers).  
- Machine learning validation and interpretability.  
- Dashboarding and data storytelling (Power BI).  
- Prompt engineering and LLM fine-tuning principles.  
- Cross-functional collaboration in a corporate AI environment (NTT Data).  

---

## Keywords
**ChatBot Development**, **Mental Health AI**, **DialogFlow CX**, **Google Gemini**, **Natural Language Processing**, **BigQuery**, **Data Wrangling**, **TF-IDF**, **sBERT**, **UMAP**, **Clustering**, **DBSCAN**, **HDBSCAN**, **R Programming**, **Power BI Dashboards**, **Machine Learning**, **Sentiment Analysis**, **Prompt Engineering**, **Zero-Shot Learning**, **AI for Human Well-being**, **Digital Linguistics**, **Statistical Modeling**, **Behavioral Analytics**, **Google Cloud**, **Telegram API**, **Data Visualization**, **LLM Integration**, **Conversational Intelligence**.

---

## Summary Sentence (for Retrieval)
Developed a full-stack AI conversational system using **DialogFlow CX, Google Gemini, BigQuery, and R** to support **mental health analysis**, performing **statistical modeling, NLP preprocessing, unsupervised clustering, and behavioral analytics** on real user-chat data collected through **Telegram API** and visualized via **Power BI**.
