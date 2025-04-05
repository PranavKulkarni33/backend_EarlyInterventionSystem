# batch_prediction.py

import pandas as pd
import re
import uuid
import os

from config import s3_client, S3_BUCKET
from rag_utils import (
    load_faiss_index_from_paths,
    get_context_records,
    format_prompt_for_prediction
)
from bedrock_integration import query_bedrock
from preprocessing import preprocess_data
from storage import upload_to_s3  # Used for uploading final report
from config import grading_scheme_map
from report_utils import generate_report_df, send_report_email



def run_batch_prediction(file_name, course_name, threshold=35, email="instructor@example.com"):
    safe_course = course_name.replace(" ", "_")
    local_csv_path = f"/tmp/{file_name}"
    preprocessed_csv_path = f"/tmp/{safe_course}_preprocessed.csv"
    index_path = f"/tmp/{safe_course}_faiss.index"

    # Download required files
    s3_client.download_file(S3_BUCKET, file_name, local_csv_path)
    s3_client.download_file(S3_BUCKET, f"{safe_course}_preprocessed.csv", preprocessed_csv_path)
    s3_client.download_file(S3_BUCKET, f"{safe_course}_faiss.index", index_path)

    # Load and normalize the batch input
    raw_df = pd.read_csv(local_csv_path)
    grading_scheme = grading_scheme_map.get(course_name, {})
    normalized_df = preprocess_data(raw_df.to_dict(orient="records"), grading_scheme)

    index, kb_df = load_faiss_index_from_paths(index_path, preprocessed_csv_path)

    predictions = []
    for _, row in normalized_df.iterrows():
        input_values = row.to_dict()
        selected_attributes = [col for col in row.keys() if col.lower() != "final"]

        try:
            context = get_context_records(kb_df, index, input_values, selected_attributes)
            prompt = format_prompt_for_prediction(input_values, selected_attributes, context, grading_scheme)
            prediction_response = query_bedrock(prompt)
            scores = extract_multi_scores(prediction_response)
        except Exception as e:
            # Log and skip row
            scores = {"Base": "-", "Optimistic": "-", "Pessimistic": "-"}

        predictions.append({
            **input_values,
            "Base": scores["Base"],
            "Optimistic": scores["Optimistic"],
            "Pessimistic": scores["Pessimistic"]
        })

    # Save the raw report
    report_df = pd.DataFrame(predictions)
    report_filename = f"{safe_course}_report_{uuid.uuid4().hex[:6]}.csv"
    report_path = f"/tmp/{report_filename}"
    report_df.to_csv(report_path, index=False)

    upload_key = f"reports/{report_filename}"
    upload_to_s3(report_path, upload_key)

    # Highlight AtRisk column
    report_df = generate_report_df(report_df.to_dict(orient="records"), threshold)

    #  Email report to instructor
    send_report_email(
        recipient_email=email,
        report_df=report_df,
        report_path=report_path,
        course_name=course_name,
        threshold=threshold
    )

    return {
        "report_url": f"s3://{S3_BUCKET}/{upload_key}",
        "at_risk_count": len(report_df[report_df["Base"].astype(str).apply(lambda x: x != '-' and float(x) < threshold)])
    }



def extract_multi_scores(response):
    """
    Extracts optimistic, base, and pessimistic scores from the model response.
    Returns a dictionary of the three scores.
    Raises a ValueError if any score is missing.
    """
    import re

    pattern = {
        "Optimistic": r"Optimistic:\s*(\d+(?:\.\d+)?)",
        "Base": r"Base:\s*(\d+(?:\.\d+)?)",
        "Pessimistic": r"Pessimistic:\s*(\d+(?:\.\d+)?)"
    }

    scores = {}
    for label, regex in pattern.items():
        match = re.search(regex, response)
        if match:
            scores[label] = float(match.group(1))
        else:
            raise ValueError(f"Missing score for {label} in response: {response}")

    return scores



