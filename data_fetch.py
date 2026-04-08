import os
import json
from datasets import load_dataset

OUTPUT_DIR = "test_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading OpenNyAI dataset from Hugging Face...")
print("Downloading OpenNyAI dataset from Hugging Face...")
try:

    HF_TOKEN = "" 
    
    dataset = load_dataset(
        "opennyaiorg/InRhetoricalRoles", 
        token=HF_TOKEN
    )
except Exception as e:
    print(f"Error connecting to Hugging Face: {e}")
    exit()

sample_cases = dataset['train'].select(range(5))

print(f"Successfully downloaded {len(sample_cases)} cases. Saving to local files...")

for idx, case in enumerate(sample_cases):
    case_id = str(case.get('id', f"case_{idx}"))
    
    safe_filename = "".join([c for c in case_id if c.isalnum() or c in (' ', '_', '-')]).strip()
    safe_filename = safe_filename.replace(" ", "_")
    
    # 2.5 Extract the text safely, no matter how deeply nested it is
    raw_data = case.get('text', case.get('data', ''))
    
    if isinstance(raw_data, dict):
        # Reach into the dictionary and grab the 'text' string
        raw_text = raw_data.get('text', str(raw_data)) 
    elif isinstance(raw_data, list):
        # If it's a list, stitch it together
        raw_text = " ".join(str(item) for item in raw_data)
    else:
        # If it's already a string, just keep it
        raw_text = str(raw_data)
        
    txt_path = os.path.join(OUTPUT_DIR, f"{safe_filename}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
        
    meta_path = os.path.join(OUTPUT_DIR, f"{safe_filename}.json")
    meta = {
        "case_number": case_id, 
        "source": "OpenNyAI_HuggingFace",
        "doc_length_chars": len(raw_text)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

print(f"\nSaved  clean test cases to the '{OUTPUT_DIR}' folder.")