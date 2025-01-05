import os
import copy
import urllib3
import pandas as pd
from minio import Minio
from dotenv import load_dotenv

ENV_PATH = "../../.env"

load_dotenv(ENV_PATH)
urllib3.disable_warnings()

ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
URL = os.getenv("MINIO_URL")
TLS = os.getenv("MINIO_TLS")

BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
INGEST_SOURCE_FILE = os.getenv("MINIO_SOURCE_FILE")
INGEST_DEST_FILE = "../../data/raw/data.csv"
INGEST_SEP = "\t"

def main():
	client = Minio(
		endpoint = URL,
		access_key = ACCESS_KEY,
		secret_key = SECRET_KEY,
		secure = TLS,
		cert_check = not TLS
	)

	if(not client.bucket_exists(BUCKET_NAME)):
		print("Bucket not found!")
		return 0
	try:
		res = client.get_object(
			bucket_name = BUCKET_NAME,
			object_name = INGEST_SOURCE_FILE
		)
		data = copy.deepcopy(res.data.decode())
		data = pd.DataFrame([row.split(",") for row in data.splitlines()])
		data.columns = data.iloc[0]
		data = data.reindex(data.index.drop(0)).reset_index(drop=True)



		data.to_csv(
			INGEST_DEST_FILE,
			sep = INGEST_SEP,
			index= False 
		)
		print("Fetching Data Success.")
	except Exception as e:
		print(str(e))

if __name__ == "__main__":
	main()
