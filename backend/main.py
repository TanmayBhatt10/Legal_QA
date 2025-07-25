from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from PyPDF2 import PdfReader
from rag import get_answer_with_rag, build_faiss_index
from utils import extract_text_from_pdf
from embedding import get_embeddings
from fastapi.templating import Jinja2Templates
import os

import tempfile

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def serve_ui(request: Request):
    return templates.TemplateResponse("legal_pt3.html", {"request": request})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

faiss_index = None
documents = []

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global faiss_index, documents
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    text = extract_text_from_pdf(tmp_path)
    documents = text.split(".")
    faiss_index = build_faiss_index(documents)
    return {"message": "File processed successfully", "chunks": len(documents)}

@app.post("/api/ask")
async def ask_question(payload: dict):
    question = payload.get("question")
    if not question:
        return JSONResponse(content={"error": "No question provided"}, status_code=400)

    answer = get_answer_with_rag(question, documents, faiss_index)
    return {"answer": answer}
