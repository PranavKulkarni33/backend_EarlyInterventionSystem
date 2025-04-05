from config import s3_client, S3_BUCKET

def upload_to_s3(file_path, s3_filename):
    """
    Uploads a file to AWS S3.
    """
    try:
        s3_client.upload_file(file_path, S3_BUCKET, s3_filename)
        print(f"Successfully uploaded {s3_filename} to {S3_BUCKET}.")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise
