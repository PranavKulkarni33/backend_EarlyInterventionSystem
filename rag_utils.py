import pandas as pd
import faiss
import os
import numpy as np
from embedding import model  # SentenceTransformer("all-MiniLM-L6-v2")
from config import grading_scheme_map as grading_scheme
import textwrap


def load_faiss_index(course_name):
    """
    Loads the FAISS index and corresponding preprocessed CSV.
    """
    safe_name = course_name.replace(" ", "_")
    index_path = f"{safe_name}_faiss.index"
    csv_path = f"{safe_name}_preprocessed.csv"

    if not os.path.exists(index_path) or not os.path.exists(csv_path):
        raise FileNotFoundError("FAISS index or preprocessed CSV not found.")

    index = faiss.read_index(index_path)
    df = pd.read_csv(csv_path)
    return index, df


def load_faiss_index_from_paths(index_path, csv_path):
    df = pd.read_csv(csv_path)
    index = faiss.read_index(index_path)
    return index, df


def input_to_sentence(input_values, selected_attributes):
    """
    Convert selected input values into a descriptive sentence for embedding.
    """
    parts = []
    for attr in selected_attributes:
        val = input_values.get(attr, "N/A")
        parts.append(f"{attr}: {val}")
    return " | ".join(parts)


def get_context_records(df, index, input_values, selected_attributes, top_k=10):
    input_str = input_to_sentence(input_values, selected_attributes)
    query_vec = model.encode([input_str]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [df.iloc[i].to_dict() for i in indices[0] if i < len(df)]


def format_prompt_for_prediction(input_values, selected_attributes, context_records, grading_weights, comment=""):
    prompt = (
        "You are an AI that predicts a student's Final exam score as a percentage (0–100).\n\n"
        "INSTRUCTION:\n"
        "Use the student's weighted scores and similar records to predict performance.\n\n"
        f"Grading Scheme (weights in %): {grading_weights}\n\n"
        "Student Attributes:\n"
    )

    for attr in selected_attributes:
        prompt += f"- {attr}: {input_values.get(attr, 'N/A')}\n"

    if comment:
        prompt += f"\nInstructor's Note: {comment}\n"

    prompt += f"\nSimilar Student Records (Top {len(context_records)}):\n"
    for i, record in enumerate(context_records, 1):
        simplified = ", ".join([f"{k}: {v}" for k, v in record.items()])
        prompt += f"{i}. {simplified}\n"

    prompt += (
        "\n✍Now predict the Final exam score in THREE cases:\n"
        "- Optimistic (if student improves)\n"
        "- Base (same performance)\n"
        "- Pessimistic (if student declines)\n\n"
        "⚠IMPORTANT:\n"
        "Your response MUST be in this **exact format**:\n"
        "Optimistic: <numeric_value>\n"
        "Base: <numeric_value>\n"
        "Pessimistic: <numeric_value>\n\n"
        "Only return these three lines. Do NOT explain or add anything else."
    )

    return prompt


