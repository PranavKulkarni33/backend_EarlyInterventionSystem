from flask import Flask, request, jsonify
import boto3
import pandas as pd
import os

app = Flask(__name__)

# AWS S3 Configuration
S3_BUCKET = "early-intervention-data"
s3_client = boto3.client("s3")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask App Running on AWS Lambda!"})


@app.route("/preprocess", methods=["POST"])
def preprocess():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract grades, weightage, class attribute, and course name
        grades = data["grades"]
        weightage = data["gradingScheme"]
        class_attribute = data["classAttribute"]
        course_name = data["courseName"].replace(" ", "_")  # Ensure filename is safe

        # Convert JSON to Pandas DataFrame
        df = pd.DataFrame(grades)

        # Handle missing values by replacing with 0
        df.fillna(0, inplace=True)

        # Normalize data based on weightage
        normalized_df = normalize_data(df, weightage)

        # Save to /tmp/ (Lambda allows writes only here)
        csv_filename = f"/tmp/{course_name}_preprocessed.csv"
        normalized_df.to_csv(csv_filename, index=False)

        # Upload to S3
        s3_client.upload_file(csv_filename, S3_BUCKET, f"{course_name}_preprocessed.csv")

        return jsonify({
            "message": f"Data preprocessed and uploaded to S3 as {course_name}_preprocessed.csv!",
            "classAttribute": class_attribute
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def normalize_data(df, weightage):
    """
    Normalize data based on the weightage provided.
    If multiple columns belong to the same entity (e.g., Quiz1, Quiz2), split the weightage equally.
    """
    normalized_df = df.copy()

    # Identify grouped entities (e.g., "Quiz1", "Quiz2" should share the "Quiz" weightage)
    entity_groups = {}
    for col in df.columns:
        for entity in weightage.keys():
            # Convert both to lowercase and remove spaces for better matching
            formatted_entity = entity.replace(" ", "").lower()
            formatted_col = col.replace(" ", "").lower()

            if formatted_col.startswith(formatted_entity):  # Match "LabTest1" with "Lab Test"
                if entity not in entity_groups:
                    entity_groups[entity] = []
                entity_groups[entity].append(col)

    # Normalize based on weightage
    for entity, columns in entity_groups.items():
        total_weight = weightage.get(entity, 0)
        split_weight = total_weight / len(columns)  # Distribute weightage evenly

        for col in columns:
            max_value = df[col].max() if df[col].max() > 0 else 1  # Avoid division by zero
            normalized_df[col] = ((df[col] / max_value) * split_weight).round(2)  # Normalize & format to 2 decimals

    return normalized_df



if __name__ == "__main__":
    app.run(debug=True)
