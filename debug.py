import os
import torch
import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("SUPABASE_URI")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
embed_model = AutoModel.from_pretrained("law-ai/InLegalBERT").to(device)

def debug_search(query_text):
    inputs = embed_tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    q_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

    conn = psycopg2.connect(DB_URI)
    register_vector(conn)
    cur = conn.cursor()

    cur.execute("""
        SELECT case_number, legal_role, chunk_text, 1 - (embedding <=> %s::vector) AS score
        FROM legal_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 10;
    """, (q_vec, q_vec))
    
    results = cur.fetchall()
    print(f"\n🔍 DEBUGGING QUERY: {query_text}")
    print("-" * 50)
    for i, row in enumerate(results):
        print(f"Rank {i+1} | Score: {row[3]:.4f} | Case: {row[0]} | Role: {row[1]}")
        print(f"Snippet: {row[2][:200]}...\n")
    cur.close()
    conn.close()

if __name__ == "__main__":
    debug_search("Why did the High Court commute the death sentences to life imprisonment in case_2?")