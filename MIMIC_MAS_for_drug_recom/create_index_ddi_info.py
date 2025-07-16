import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

import json
import logging
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
from tqdm import tqdm
import torch
device = torch.device("cuda:0")  

print(f"Using device: {device}")


def generate_embeddings(texts, model, tokenizer, max_length=512):
    """Generate embeddings for a list of texts using a pre-trained model."""
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)

def load_chunks_to_memory(file_path):
    """Preload all chunks from a file into memory as a dictionary."""
    chunk_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                chunk_map[record["id"]] = record["chunk"]
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Error parsing line: {line.strip()} - {e}")
    return chunk_map

def create_faiss_index(folder_path, index_path, model, tokenizer, batch_size=128):
    """Create a FAISS index from all JSONL files in a folder."""
    index = faiss.IndexFlatL2(768)  # Adjust dimension if necessary
    id_to_file_map = {}
    for file_path in tqdm([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jsonl")], desc="Processing files"):
            chunks = []
            ids = []
            with open(file_path, "r", encoding="utf-8") as f:
                 for line_number, line in enumerate(f):
                    try:
                        record = json.loads(line)
                        chunks.append(record["content"])
                        ids.append(record["id"])
                        # ids.append(record["id"])
                        #还有title contents 两项，这里似乎可以不用
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(f"Failed to parse line in {file_path}: {e}")

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                embeddings = generate_embeddings(batch_chunks, model, tokenizer)
                index.add(embeddings)
                id_to_file_map.update({ids[j]: file_path for j in range(i, i + len(batch_chunks))})
    # for file_path in folders:
    #     print("现在转换的是：", file_path)
        

    faiss.write_index(index, index_path)
    with open(index_path + ".map", "w", encoding="utf-8") as f:
        json.dump(id_to_file_map, f)

    logging.info("Index creation complete.")

def query_faiss_index(query_text, index_path, model, tokenizer, top_k=5):
    """Query a FAISS index and return the top-k results."""
    index = faiss.read_index(index_path)
    with open(index_path + ".map", "r", encoding="utf-8") as f:
        id_to_file_map = json.load(f)

    query_embedding = generate_embeddings([query_text], model, tokenizer)
    _, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        file_name = id_to_file_map[str(idx)]
        chunk_map = load_chunks_to_memory(file_name)
        results.append({"id": idx, "content": chunk_map.get(idx)})

    return results

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    article_model_name = "ncbi/MedCPT-Article-Encoder"
    article_tokenizer = AutoTokenizer.from_pretrained(article_model_name)
    article_model = AutoModel.from_pretrained(article_model_name)
    article_model.to(device)

    # query_model_name = "ncbi/MedCPT-Query-Encoder"
    # query_model = AutoModel.from_pretrained(query_model_name)
    # query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
    folders = ["/huggingface/hub/datasets--MedRAG--pubmed/snapshots/33da3593d5756bc04c8909f170003c0b14197957/chunk", 
               "/huggingface/hub/datasets--MedRAG--textbooks/snapshots/9c72838920a1323ffa867467d3f7aa7b36b0f994/chunk", 
               "/huggingface/hub/datasets--MedRAG--wikipedia/snapshots/d76b9ad82135e352235d17e75921c49b68fd07b2/chunk", 
               "llm/MedRAG/corpus/statpearls/chunk",
               "llm/Code-for-DDI/RAGents4DDI/data4rag/ddi_info"] 
    folder_path = "/huggingface/hub/datasets--MedRAG--textbooks/snapshots/9c72838920a1323ffa867467d3f7aa7b36b0f994/chunk"
    index_path = "llm/Code-for-DDI/RAGents4DDI/faiss/faiss_index_ddi_info.idx"

    # Create FAISS index
    create_faiss_index(folders[4], index_path, article_model, article_tokenizer, batch_size=2048)
    print('done')
    # Query the index
    # query = "What is the capital of France?"
    # results = query_faiss_index(query, index_path, query_model, query_tokenizer, top_k=3)

    # for result in results:
    #     print(result)
