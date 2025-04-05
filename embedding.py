# embedding.py

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def search_index(index_path, csv_path, query, top_k=3):
    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Embed the query
    query_vec = model.encode([query])

    # Perform search
    distances, indices = index.search(query_vec, top_k)

    # Retrieve top rows as context
    rows = [df.iloc[i].to_dict() for i in indices[0] if i < len(df)]

    return rows

def index_csv_file(csv_path, course_name):
    """
    Converts the preprocessed CSV into a FAISS index using sentence embeddings.
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_path)

        # Convert each row to a string
        texts = df.astype(str).apply(lambda row: " | ".join(row.values), axis=1).tolist()

        # Load sentence transformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts)

        # Convert to numpy float32
        embeddings = np.array(embeddings).astype("float32")

        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # Save index to file
        index_path = f"/tmp/{course_name}_faiss.index"
        faiss.write_index(index, index_path)

        return index_path

    except Exception as e:
        print(f"Error in indexing: {e}")
        raise
