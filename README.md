# AI-Powered-Legal-Assistant

AI-Powered Legal Assistant is a Python-based **RAG (Retrieval-Augmented Generation)** project for Indian legal research.  
It helps users ask legal questions in natural language, retrieves relevant precedent chunks from a vector database, and generates a final response with Gemini.

## Why this project exists

Indian legal research is often difficult for students and early-career practitioners because:
- legal material is large and fragmented,
- keyword search can miss semantically similar precedents,
- high-quality legal databases can be expensive.

This project aims to reduce that friction by combining legal-domain embeddings with LLM-based answer generation.

## Architecture and workflow

The system follows a classic RAG flow:

1. **Ingestion (`data_ingestion.py`)**
   - Reads `.txt` files from `test_data/` (+ optional `.json` metadata per file).
   - Splits text into sentences with spaCy.
   - Classifies each sentence into legal rhetorical roles using `Ansh-Singhal/inlegalbert-legalseg`.
   - Builds overlapping chunks per role.
   - Creates embeddings using `law-ai/InLegalBERT`.
   - Uploads records into PostgreSQL/Supabase table `legal_chunks` (pgvector).

2. **Vector retrieval (`vector_search.py` and API retrieval in `api.py`)**
   - User question is embedded with the same InLegalBERT embedding model.
   - Similarity search is run in SQL (`ORDER BY embedding <=> query_vector`).
   - Top matching chunks are returned from `legal_chunks`.

3. **Response generation (`api.py`)**
   - Flask endpoint receives the query.
   - Retrieved precedent context is formatted into a prompt.
   - Gemini (`gemini-2.5-flash`) generates the final answer.

## Main files and modules

- `api.py`  
  Flask API (`POST /api/search`) for end-to-end retrieval + Gemini answer generation.

- `data_ingestion.py`  
  Core ingestion pipeline: rhetorical-role classification, chunking, embedding, DB insert.

- `vector_search.py`  
  Standalone semantic retrieval test script (returns top-5 chunks with scores).

- `data_fetch.py`  
  Utility script to download sample OpenNyAI data and write `.txt/.json` files to `test_data/`.

- `llm_ingestion.py`  
  Standalone script that retrieves top chunks and asks Gemini to synthesize a response (non-API flow).

- `test.py`, `debug.py`  
  Local experimentation scripts for retrieval + Gemini prompting.

- `database_test.py`  
  Simple PostgreSQL connectivity check script.

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Access to a PostgreSQL/Supabase database with `pgvector` enabled
- Google Gemini API key
- Enough RAM/CPU (or GPU) to load transformer models

## Installation

```bash
cd /home/runner/work/AI-Powered-Legal-Assistant/AI-Powered-Legal-Assistant
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install flask flask-cors python-dotenv torch transformers spacy psycopg2-binary pgvector google-genai datasets
python -m spacy download en_core_web_sm
```

## Environment variables

Create a `.env` file in the project root:

```env
SUPABASE_URI=postgresql://<user>:<password>@<host>:<port>/<db>
GEMINI_API_KEY=<your_gemini_api_key>
```

Optional for dataset fetch:

```env
HF_TOKEN=<huggingface_token_if_required>
```

## Data ingestion

1. Put legal case files in `test_data/` as `.txt` files.  
   Optionally add matching `.json` files with metadata (for example `case_number`).
2. Run ingestion:

```bash
python data_ingestion.py
```

This script creates role-aware chunks, embeddings, and inserts into `legal_chunks`.

## Run the API locally

```bash
python api.py
```

By default, Flask runs on `http://0.0.0.0:5000`.

## API usage example

### Request

```bash
curl -X POST http://127.0.0.1:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What factors are considered when evaluating reliability of a dying declaration?"}'
```

### Response shape

```json
{
  "answer": "....",
  "status": "success"
}
```

## Vector search (high-level)

- Text chunks and user query are encoded into 768-d vectors via InLegalBERT.
- PostgreSQL `pgvector` uses `<=>` distance operator for nearest-neighbor ranking.
- Lower distance means greater semantic similarity.

## Project structure

```text
AI-Powered-Legal-Assistant/
├── api.py
├── data_fetch.py
├── data_ingestion.py
├── vector_search.py
├── llm_ingestion.py
├── test.py
├── debug.py
├── database_test.py
├── test_data/
└── README.md
```

## Caveats and current limitations

- The project depends on external services (Supabase/PostgreSQL + Gemini API) and will not run fully offline.
- Expected DB schema/table creation SQL is not included in this repository; `legal_chunks` must already exist.
- In ingestion, `decision_date` and `judge_name` are currently inserted as `"Unknown"`.
- Model loading is done at startup and can be slow/heavy on low-resource machines.
- API prompt allows a Gemini fallback “general knowledge mode,” so some answers may not be strictly grounded in retrieved chunks.
- There is no built-in authentication/rate limiting in the Flask API.
