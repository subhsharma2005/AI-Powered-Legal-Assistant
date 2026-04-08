import os
import torch
import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel
from google import genai
from google.genai import types
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app) 
load_dotenv()

DB_URI = os.getenv("SUPABASE_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DB_URI or not GEMINI_API_KEY:
    raise ValueError("Missing SUPABASE_URI or GEMINI_API_KEY in environment variables.")

client = genai.Client(api_key=GEMINI_API_KEY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Starting server on {device.type.upper()}...")
print("Loading InLegalBERT into memory. This might take a few seconds...")
embed_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
embed_model = AutoModel.from_pretrained("law-ai/InLegalBERT").to(device)
print("Models loaded successfully. API is ready!")

def embed_question(text):
    """Turns the user's question into a mathematical vector."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

@app.route('/api/search', methods=['POST'])
def search_legal_ai():
    data = request.get_json()
    user_question = data.get('query')

    if not user_question:
        return jsonify({"error": "No query provided"}), 400

    try:
        q_vec = embed_question(user_question)
        
        conn = psycopg2.connect(DB_URI)
        register_vector(conn)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT case_number, legal_role, chunk_text
            FROM legal_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 80;
        """, (q_vec,))
        
        results = cur.fetchall()
        cur.close()
        conn.close()

        if not results:
            return jsonify({"answer": "I could not find any relevant precedents in the database to answer this question."}), 200

        context_block = ""
        for i, row in enumerate(results):
            context_block += f"\n--- Precedent {i+1} (Case: {row[0]} | Role: {row[1]}) ---\n{row[2]}\n"

        system_prompt = """
        You are an expert Indian Legal AI Assistant and a mentor for law students. 
        You have two modes of operation: 'Precedent Mode' and 'General Knowledge Mode'.

        --- MODE 1: PRECEDENT MODE (Primary) ---
        Use this mode if the provided 'Precedent Context' contains information relevant to the user's question. 
        Your answer must be grounded ONLY in the context. Cite specific case IDs like (case_2) or (case_116).
        
        FORMAT FOR PRECEDENT MODE:
        🏛️ Court: [Extract Court Name]
        ⚖️ Parties: [Extract Appellant vs. Respondent]
        📖 Key Provisions: [List Acts/Sections, e.g., Section 302 IPC]
        💡 Legal Principle (Ratio): [1-sentence core legal reasoning]
        ---------------------------------------------------------
        [Detailed Answer with citations to the Precedents]

        --- MODE 2: GENERAL KNOWLEDGE MODE (Fallback) ---
        Use this mode ONLY if the 'Precedent Context' is empty or does not contain the specific answer (e.g., famous cases like Kesavananda Bharati or general definitions). 
        Do NOT say "I do not have sufficient information." Instead, use your internal legal training to provide a high-quality educational answer.

        FORMAT FOR GENERAL KNOWLEDGE MODE:
        🧠 **General Legal Knowledge (Not found in specific database precedents)**
        ---------------------------------------------------------
        [Provide a detailed, accurate legal explanation or historical case summary here.]

        --- CRITICAL RULES ---
        1. DECIDE FIRST: Check the context. If the answer is there, use Mode 1. If not, use Mode 2. NEVER combine the two formats.
        2. WHITESPACE: Always use double newlines between sections. This is vital for the React frontend's 'whitespace-pre-wrap' styling.
        3. NO REPETITION: Output the metadata/header block EXACTLY ONCE.
        4. CITATIONS: In Mode 1, always mention which 'Case' or 'Precedent' number you are referring to.
        5. TONE: Maintain a professional, mentorship-style tone for a law student.
        """

        prompt = f"{system_prompt}\n\nUSER QUESTION: {user_question}\n\nPRECEDENT CONTEXT:\n{context_block}"

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3) 
        )

        return jsonify({
            "answer": response.text,
            "status": "success"
        }), 200

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)