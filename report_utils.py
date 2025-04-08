import pandas as pd
import boto3
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from config import S3_BUCKET

ses_client = boto3.client("ses", region_name="us-east-1")
SENDER_EMAIL = "earlyinterventionsystematyork@gmail.com"


def generate_report_df(records, threshold):
    df = pd.DataFrame(records)

    # Replace missing or failed predictions with NaN
    df["Base"] = pd.to_numeric(df["Base"], errors='coerce')  # invalid strings like '-' → NaN
    df["AtRisk"] = df["Base"] < threshold  # only valid numbers will be checked
    df["Base"] = df["Base"].fillna("-")  # keep display consistent
    return df



def send_report_email(recipient_email, report_df, report_path, course_name, threshold=35):
    at_risk_df = report_df[report_df["AtRisk"] == True]
    at_risk_count = len(at_risk_df)

    summary = (
        f"Hi,\n\nPlease find attached the early intervention report for course: {course_name}.\n"
        f"Number of students at risk (below threshold): {at_risk_count}\n\n"
        f"Regards,\nEarly Intervention System"
    )

    msg = MIMEMultipart()
    msg["Subject"] = f"Early Intervention Report - {course_name}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg.attach(MIMEText(summary, "plain"))

    with open(report_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(report_path))
        part.add_header("Content-Disposition", "attachment", filename=os.path.basename(report_path))
        msg.attach(part)

    try:
        response = ses_client.send_raw_email(
            Source=SENDER_EMAIL,
            Destinations=[recipient_email],
            RawMessage={"Data": msg.as_string()}
        )
        print("Email sent!", response["MessageId"])
    except Exception as e:
        print("SES email error:", e)
