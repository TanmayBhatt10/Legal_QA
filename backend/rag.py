import faiss
import numpy as np
from embedding import get_embeddings
import openai
import os

def build_faiss_index(chunks):
    embeddings = get_embeddings(chunks)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

def get_answer_with_rag(question, chunks, index, top_k=3):
    question_embedding = get_embeddings([question])[0]
    D, I = index.search(np.array([question_embedding]).astype("float32"), top_k)
    context = "\n\n".join([chunks[i] for i in I[0]])
    return ask_openai(question, context)

def ask_openai(question, context):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"""Context:
{context}

Question: {question}
Answer:"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]
