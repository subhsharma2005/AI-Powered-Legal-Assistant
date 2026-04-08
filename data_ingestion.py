import os
import json
import spacy
import torch
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("SUPABASE_URI")
INPUT_FOLDER = "test_data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading Spacy and InLegalBERT models...")
nlp = spacy.load("en_core_web_sm") 

seg_tokenizer = AutoTokenizer.from_pretrained("Ansh-Singhal/inlegalbert-legalseg")
seg_model = AutoModelForSequenceClassification.from_pretrained("Ansh-Singhal/inlegalbert-legalseg").to(device)

embed_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
embed_model = AutoModel.from_pretrained("law-ai/InLegalBERT").to(device)

id2label = {0: "Facts", 1: "Issue", 2: "Arguments of Petitioner", 3: "Arguments of Respondent", 4: "Reasoning", 5: "Decision", 6: "Precedent"}


def get_embedding(text):
    """Turns text into a 768-dimensional math vector."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

def sliding_window_chunk(text, max_words=250, overlap=50):
    """Slices long text into manageable chunks with overlap to retain context."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def run_ingestion():
    if not DB_URI:
        print("❌ Error: SUPABASE_URI not found in .env file.")
        return

    conn = psycopg2.connect(DB_URI)
    register_vector(conn)
    cur = conn.cursor()
    
    print("Connected to Supabase successfully.")
    
    txt_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]
    batch_data = []

    print(f"\nProcessing {len(txt_files)} files...")
    
    for filename in txt_files:
        base = filename.replace('.txt', '')
        print(f"  -> Processing: {base}")
        
        with open(os.path.join(INPUT_FOLDER, filename), 'r', encoding='utf-8') as f: 
            text = f.read()
        
        meta_path = os.path.join(INPUT_FOLDER, f"{base}.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f: 
                meta = json.load(f)

        buckets = {role: [] for role in id2label.values()}
        for sent in nlp(text).sents:
            if not sent.text.strip(): continue
            inputs = seg_tokenizer(sent.text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                role = id2label[torch.argmax(seg_model(**inputs).logits).item()]
            buckets[role].append(sent.text)

        for role, sents in buckets.items():
            if not sents: continue
            
            full_role_text = " ".join(sents)
            smaller_chunks = sliding_window_chunk(full_role_text)
            
            for idx, chunk in enumerate(smaller_chunks):
                vector = get_embedding(chunk)
                
                case_num = meta.get('case_number', 'Unknown Case')
                enriched_chunk = f"Case: {case_num}. Role: {role}. Text: {chunk}"
                
                batch_data.append((
                    f"{base}_{role}_{idx}",
                    case_num,
                    "Unknown", # Date
                    "Unknown", # Judge
                    role,
                    enriched_chunk,
                    vector
                ))

    print(f"\nUploading {len(batch_data)} enriched chunks to Supabase in batches of 50...")
    
    for i in range(0, len(batch_data), 50):
        mini_batch = batch_data[i:i + 50]
        try:
            execute_values(cur, """
                INSERT INTO legal_chunks 
                (chunk_id, case_number, decision_date, judge_name, legal_role, chunk_text, embedding) 
                VALUES %s
                ON CONFLICT (chunk_id) DO NOTHING; -- Prevents duplicate errors
            """, mini_batch)
            conn.commit()
            print(f"  ✅ Uploaded chunks {i} to {i + len(mini_batch)}")
        except Exception as e:
            print(f"  ❌ Batch failed at chunk {i}: {e}")
            conn.rollback()
            
    cur.close()
    conn.close()
    print("\n🎉 DATA INGESTION COMPLETE! Your database is now populated with high-visibility vectors.")

if __name__ == "__main__":
    run_ingestion()