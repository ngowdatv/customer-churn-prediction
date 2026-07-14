import boto3

s3 = boto3.client('s3')

bucket_name = "customer-churn-bucket-12345"
file_name = "model.pkl"

s3.download_file(bucket_name, file_name, "downloaded_model.pkl")

print("Downloaded from S3 ✅")