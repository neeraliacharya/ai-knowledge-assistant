import os
from app.services.ingestion import extract_text_from_pdf

def load_documents(folder_path="documents"):

    documents = []

    for file in os.listdir(folder_path):

        if file.lower().endswith(".pdf"):

            path = os.path.join(folder_path, file)

            text = extract_text_from_pdf(path)

            documents.append({
                "text": text,
                "source": file
            })

    return documents