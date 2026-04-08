import os
import torch
import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("SUPABASE_URI")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Waking up InLegalBERT for final generation...")
embed_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
embed_model = AutoModel.from_pretrained("law-ai/InLegalBERT").to(device)

def embed_question(text):
    """Turns the user's question into a mathematical vector."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

def ask_legal_ai(user_question):
    print(f"\n[1] Vectorizing Question: '{user_question}'")
    q_vec = embed_question(user_question)
    
    print("[2] Searching Supabase (Pulling top 30 chunks to avoid misses)...")
    conn = psycopg2.connect(DB_URI)
    register_vector(conn)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT case_number, legal_role, chunk_text
        FROM legal_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 30;
    """, (q_vec,))
    
    results = cur.fetchall()
    cur.close()
    conn.close()

    if not results:
        print("❌ No results found in the database.")
        return

    print("\n--- SCOOPED CONTEXT FOR GEMINI ---")
    context_block = ""
    for i, row in enumerate(results):
        case_num = row[0]
        role = row[1]
        text = row[2]
        
        print(f"Rank {i+1}: {case_num} | Role: {role}")
        
        context_block += f"\n--- Precedent {i+1} (Case: {case_num} | Role: {role}) ---\n{text}\n"
    print("----------------------------------\n")

    print("[3] Asking Gemini to synthesize the final answer...")
    
    system_prompt = """
    You are an expert Indian Legal AI Assistant and a mentor for law students. 
    Your ONLY job is to answer the user's question using the provided 'Precedent Context'.

    CRITICAL FORMATTING REQUIREMENT:
    Before answering the user's question, you MUST scan the context and extract the legal metadata to help law students study. Start your final output exactly like this:

     Court: [Extract the Court Name, e.g., High Court of Karnataka]
     Parties: [Extract Appellant/Petitioner] vs. [Extract Respondent/State]
     Key Provisions: [List the specific Acts, Sections, or Articles discussed, e.g., Section 498A IPC, Section 306 IPC]
     Legal Principle (Ratio): [Provide a 1-sentence summary of the core legal rule or reasoning applied by the judge in this specific text]
     Case id: [provide the name of the file]

    ---------------------------------------------------------
    [Your detailed, cited answer to the user's question goes here. Explain the logic clearly for a law student.]

    RULES:
    1. If a specific piece of metadata (like the Court or Section) is not in the context, write "Not explicitly mentioned in retrieved context".
    2. Ignore context that is completely unrelated to the question.
    3. If the relevant answer is not in the context, explicitly state: "I do not have sufficient information in the provided precedents to answer this."
    4. Do NOT use outside knowledge. Rely ONLY on the provided context blocks.
    """

    prompt = f"""
    {system_prompt}
    
    USER QUESTION: {user_question}
    
    PRECEDENT CONTEXT:
    {context_block}
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1) 
    )

    print("\n" + "=" * 80)
    print("⚖️ FINAL LEGAL ANSWER:")
    print("=" * 80)
    print(response.text)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_query = "What factors do the courts consider when determining if a dying declaration is reliable and voluntarily made without tutoring?"
    ask_legal_ai(test_query)