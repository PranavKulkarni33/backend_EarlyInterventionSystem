import pandas as pd
import faiss
import os
import numpy as np
from embedding import model  # SentenceTransformer("all-MiniLM-L6-v2")
from config import grading_scheme_map as grading_scheme


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


def get_context_records(df, index, input_values, selected_attributes, top_k=3):
    input_str = input_to_sentence(input_values, selected_attributes)
    query_vec = model.encode([input_str]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [df.iloc[i].to_dict() for i in indices[0] if i < len(df)]


def format_prompt_for_prediction(input_values, selected_attributes, context_records, grading_weights, comment=""):
    prompt = (
        "You are an academic performance predictor. Use the grading scheme and context of similar students "
        "to predict the Final exam score as a percentage (0-100).\n\n"
        f"Grading Scheme: {grading_weights}\n"
        "Student Attributes:\n"
    )
    for attr in selected_attributes:
        val = input_values.get(attr, "N/A")
        prompt += f"- {attr}: {val}\n"

    if comment:
        prompt += f"Instructor's Comment: {comment}\n"

    prompt += "\nSimilar Student Records:\n"
    for i, record in enumerate(context_records, 1):
        record_str = ", ".join([f"{k}: {v}" for k, v in record.items()])
        prompt += f"{i}. {record_str}\n"

    prompt += (
        "\nNow predict the student's Final exam score in three cases:\n"
        "- Optimistic (if the student improves):\n"
        "- Base (if the student maintains similar performance):\n"
        "- Pessimistic (if the student does worse):\n\n"
        "You MUST respond ONLY in this exact format with numeric values:\n"
        "Optimistic: <score>\nBase: <score>\nPessimistic: <score>\n"
        "Do not include explanations or ranges. Return only these three values."
    )

    return prompt

