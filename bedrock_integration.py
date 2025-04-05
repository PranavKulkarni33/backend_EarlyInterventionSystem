import boto3
import json
from config import S3_BUCKET

bedrock_client = boto3.client("bedrock-runtime")

def generate_response_from_context(query, context_rows, model_id="amazon.titan-text-express-v1"):
    context_str = "\n".join([str(row) for row in context_rows])

    prompt = f"""You are an AI assistant helping a course instructor.

Context:
{context_str}

Question:
{query}

Answer:"""

    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 100,
            "temperature": 0.7,
            "topP": 0.9
        }
    }

    try:
        response = bedrock_client.invoke_model(
            body=json.dumps(payload),
            modelId=model_id,
            contentType="application/json",
            accept="application/json"
        )
        raw = response["body"].read().decode("utf-8")
        print("Raw Bedrock response:", raw)

        data = json.loads(raw)
        return data["results"][0]["outputText"]
    except Exception as e:
        print(f"Bedrock API error: {e}")
        return None



def query_bedrock(prompt, model_id="amazon.titan-text-express-v1"):
    """
    Sends a prompt to Bedrock and returns the model's response.
    """
    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 100,
            "temperature": 0.7,
            "topP": 0.9
        }
    }

    try:
        response = bedrock_client.invoke_model(
            body=json.dumps(payload),
            modelId=model_id,
            contentType="application/json",
            accept="application/json"
        )
        raw = response["body"].read().decode("utf-8")
        print("Raw Bedrock response:", raw)

        data = json.loads(raw)
        output = data.get("results", [{}])[0].get("outputText")
        if not output:
            raise ValueError("Missing 'outputText' in Bedrock response")

        return output
    except Exception as e:
        print(f"Bedrock API error: {e}")
        return "outputText missing in response"

