# AI-Powered Legal Assistant
A Python-based Retrieval-Augmented Generation (RAG) system for Indian legal question answering.
## Overview
This project ingests legal case text, breaks it into role-aware chunks, stores vector embeddings in PostgreSQL/pgvector (via Supabase), retrieves semantically similar chunks for a user query, and uses Google Gemini to generate a final answer.
It is designed for legal research and study workflows where users ask legal questions in natural language and receive responses grounded in retrieved precedent context.
---
## Architecture / Workflow
1. **Ingestion (`data_ingestion.py`)**
   - Reads `.txt` legal documents from `test_data/`
   - Loads optional metadata from matching `.json` files
2. **Labeling + Chunking**
   - Uses spaCy sentence segmentation
   - Classifies each sentence into legal rhetorical roles using `Ansh-Singhal/inlegalbert-legalseg`
   - Buckets by role and applies sliding-window chunking with overlap
3. **Embedding**
   - Embeds each chunk using `law-ai/InLegalBERT`
4. **Vector Storage**
   - Writes enriched chunks to `legal_chunks` in PostgreSQL with pgvector
5. **Retrieval (`api.py` / `vector_search.py`)**
   - Embeds user query with `law-ai/InLegalBERT`
   - Performs vector similarity search in `legal_chunks`
6. **Answer Generation (`api.py`)**
   - Builds context from retrieved chunks
   - Calls Gemini (`gemini-2.5-flash`) to generate the final response
---
## Repository Structure
- `api.py` - Flask API (`POST /api/search`) for retrieval + Gemini answer generation
- `data_ingestion.py` - end-to-end ingestion pipeline (read -> label -> chunk -> embed -> store)
- `vector_search.py` - local retrieval test script (top-k nearest chunks)
- `llm_ingestion.py` - CLI-style retrieval + Gemini synthesis script
- `data_fetch.py` - helper script to download and save sample legal data into `test_data/`
- `test_data/` - local corpus input directory (`.txt` and optional `.json` metadata)
- `debug.py`, `test.py`, `database_test.py` - utility/debug scripts
---
## Tech Stack
- Python
- Flask + Flask-CORS
- PyTorch
- spaCy (`en_core_web_sm`)
- Hugging Face Transformers
- PostgreSQL + pgvector (Supabase)
- Google Gemini (`google-genai`)
---
## Prerequisites
- Python 3.x
- PostgreSQL database with pgvector enabled (or Supabase Postgres with pgvector)
- Access keys for required external services
- Enough memory/compute to load transformer models (GPU optional; CPU supported)
---
## Installation & Setup
```bash
git clone https://github.com/subhsharma2005/AI-Powered-Legal-Assistant.git
cd AI-Powered-Legal-Assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install flask flask-cors torch transformers spacy psycopg2-binary pgvector python-dotenv google-genai datasets
python -m spacy download en_core_web_sm
```
> If you maintain dependencies via a lockfile/requirements file in your environment, prefer that for reproducibility.
---
## Required Environment Variables
Create a `.env` file in the repository root:
```env
SUPABASE_URI=postgresql://<user>:<password>@<host>:<port>/<db>
GEMINI_API_KEY=<your_gemini_api_key>
```
- `SUPABASE_URI` - used by ingestion and retrieval scripts to connect to Postgres
- `GEMINI_API_KEY` - required by `api.py` and `llm_ingestion.py` for response generation
---
## Data Ingestion
1. Put legal documents in `test_data/` as `.txt` files.
2. (Optional) Add matching metadata `.json` files with at least `case_number`.
3. Run:
```bash
python data_ingestion.py
```
This populates the `legal_chunks` table with role-labeled, embedded chunks.
---
## Run API Locally
```bash
python api.py
```
Server starts on:
- `http://0.0.0.0:5000`
- Endpoint: `POST /api/search`
---
## API Usage Example
### Request
```bash
curl -X POST http://127.0.0.1:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What factors make a dying declaration reliable?"}'
```
### Response (example shape)
```json
{
  "answer": "...Gemini-generated legal answer...",
  "status": "success"
}
```
Error example (missing query):
```json
{
  "error": "No query provided"
}
```
---
