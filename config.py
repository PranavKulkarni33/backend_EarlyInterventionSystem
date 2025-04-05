import boto3

# AWS S3 Configuration
S3_BUCKET = "early-intervention-data"
s3_client = boto3.client("s3")
# Global map to hold grading schemes for each course
grading_scheme_map = {}