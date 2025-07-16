import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(texts):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [e["embedding"] for e in response["data"]]