from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from preprocessing import preprocess_data
from storage import upload_to_s3
from embedding import index_csv_file, search_index
from config import S3_BUCKET, s3_client
from bedrock_integration import generate_response_from_context, query_bedrock
from rag_utils import (
    load_faiss_index_from_paths,
    get_context_records,
    format_prompt_for_prediction
)
from config import grading_scheme_map
from batch_prediction import extract_multi_scores
import json
import tempfile
from config import S3_BUCKET, s3_client


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask App Running!"})


@app.route("/preprocess", methods=["POST"])
def preprocess():
    try:
        data = request.get_json()
        grades = data["grades"]
        weightage = data["gradingScheme"]
        class_attribute = data["classAttribute"]
        course_name = data["courseName"].replace(" ", "_")

        # Preprocess grades
        normalized_df = preprocess_data(grades, weightage)

        # Save preprocessed CSV
        csv_filename = f"/tmp/{course_name}_preprocessed.csv"
        normalized_df.to_csv(csv_filename, index=False)
        upload_to_s3(csv_filename, f"{course_name}_preprocessed.csv")

        # Save FAISS index
        index_path = index_csv_file(csv_filename, course_name)
        upload_to_s3(index_path, f"{course_name}_faiss.index")

        # ✅ Extract column list and exclude class attribute if needed
        column_list = list(normalized_df.columns)

        # Save metadata (✅ added "columns")
        metadata = {
            "courseName": course_name,
            "gradingScheme": weightage,
            "classAttribute": class_attribute,
            "columns": column_list,
            "numRecords": len(normalized_df),
            "numFeatures": len(normalized_df.columns)
        }

        meta_path = f"/tmp/{course_name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        upload_to_s3(meta_path, f"{course_name}_metadata.json")

        return jsonify({
            "message": "Preprocessing successful. CSV, index, and metadata uploaded.",
            "csv_file": f"{course_name}_preprocessed.csv",
            "index_file": f"{course_name}_faiss.index",
            "metadata_file": f"{course_name}_metadata.json"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route("/models", methods=["GET"])
def list_models():
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET)
        files = response.get("Contents", [])

        metadata_files = [f["Key"] for f in files if f["Key"].endswith("_metadata.json")]

        models = {}
        for key in metadata_files:
            course_name = key.replace("_metadata.json", "")
            local_path = f"/tmp/{key}"
            s3_client.download_file(S3_BUCKET, key, local_path)

            with open(local_path, "r") as f:
                metadata = json.load(f)

            models[course_name] = metadata

        return jsonify(models)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route("/predict-individual", methods=["POST"])
def predict_individual():
    try:
        data = request.get_json()
        course = data["courseName"].replace(" ", "_")
        selected_attributes = data["selectedAttributes"]
        input_values = data["inputValues"]
        comment = data.get("comment", "")

        csv_path = f"/tmp/{course}_preprocessed.csv"
        index_path = f"/tmp/{course}_faiss.index"

        s3_client.download_file(S3_BUCKET, f"{course}_preprocessed.csv", csv_path)
        s3_client.download_file(S3_BUCKET, f"{course}_faiss.index", index_path)

        index, df = load_faiss_index_from_paths(index_path, csv_path)
        similar_records = get_context_records(df, index, input_values, selected_attributes)

        grading_scheme = grading_scheme_map.get(course, {})
        prompt = format_prompt_for_prediction(input_values, selected_attributes, similar_records, grading_scheme, comment)
        raw_response = query_bedrock(prompt)
        scores = extract_multi_scores(raw_response)

        return jsonify({
            "input": input_values,
            "context": similar_records,
            "prompt": prompt,
            "prediction": {
                "Optimistic": scores["Optimistic"],
                "Base": scores["Base"],
                "Pessimistic": scores["Pessimistic"]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    try:
        file = request.files["file"]
        course = request.form["courseName"]
        threshold = float(request.form.get("threshold", 35))
        email = request.form.get("email", "instructor@example.com")

        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)

        s3_client.upload_file(temp_path, S3_BUCKET, file.filename)

        from batch_prediction import run_batch_prediction
        result = run_batch_prediction(file.filename, course, threshold, email)

        print("Batch Prediction Result (final):", result)

        if not result or "report_url" not in result or "at_risk_count" not in result:
            raise ValueError("Batch result missing expected keys")

        return jsonify({
            "message": "Batch prediction completed.",
            "reportUrl": result["report_url"],
            "atRiskCount": result["at_risk_count"]
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
