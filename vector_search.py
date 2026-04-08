import os
import torch
import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("SUPABASE_URI")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Waking up InLegalBERT for search...")
embed_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
embed_model = AutoModel.from_pretrained("law-ai/InLegalBERT").to(device)

def embed_question(text):
    """Turns your text question into a 768-dimensional vector."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

def test_vector_search(query_text):
    print(f"\n🔍 QUESTION: '{query_text}'")
    print("Converting to vector and searching Supabase...")
    
    q_vec = embed_question(query_text)
    
    conn = psycopg2.connect(DB_URI)
    register_vector(conn)
    cur = conn.cursor()
    

    cur.execute("""
        SELECT case_number, legal_role, chunk_text, 1 - (embedding <=> %s::vector) AS similarity_score
        FROM legal_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 5;
    """, (q_vec, q_vec))
    
    results = cur.fetchall()
    cur.close()
    conn.close()

    if not results:
        print("❌ No results found. Is the database empty?")
        return

    print("\n" + "="*80)
    print("🏆 TOP 5 RETRIEVED CHUNKS")
    print("="*80)
    
    for i, row in enumerate(results):
        case_num = row[0]
        role = row[1]
        text = row[2]
        score = row[3]
        
        print(f"\n🥇 RANK {i+1} | Score: {score:.4f} | Case: {case_num} | Role: {role}")
        print("-" * 80)
        print(f"{text[:1000]}...\n")

if __name__ == "__main__":
    test_question = "On what date did the incident involving Asha Hemant Chauriwal occur, and what was the medical cause of her death as stated in the post-mortem report?"
    
    test_vector_search(test_question)