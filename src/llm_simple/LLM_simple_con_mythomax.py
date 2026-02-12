import fitz  # PyMuPDF para extraer texto de PDFs
from sentence_transformers import SentenceTransformer
import faiss
import requests
import torch

# 1. Extraer texto del PDF en bloques largos
def extract_text_from_pdf(pdf_path, max_chunk_chars=600):
    doc = fitz.open(pdf_path)
    all_chunks = []
    for page in doc:
        text = page.get_text("text")
        paragraphs = text.split("\n\n")
        chunk = ""
        for para in paragraphs:
            if len(chunk) + len(para) < max_chunk_chars:
                chunk += " " + para.strip()
            else:
                all_chunks.append(chunk.strip())
                chunk = para.strip()
        if chunk:
            all_chunks.append(chunk.strip())
    return all_chunks

# 2. Procesar el PDF
pdf_texts = extract_text_from_pdf("ley.pdf")

# 3. Cargar modelo de embeddings (mejor para español)
model = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda" if torch.cuda.is_available() else "cpu")
embeddings = model.encode(pdf_texts)

# 4. Crear índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 5. Buscar fragmentos más relevantes
def search_relevant_text(query, top_k=4):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    matched_texts = [pdf_texts[i] for i in indices[0]]
    print("🧠 Texto relevante entregado al modelo:\n")
    for i, fragment in enumerate(matched_texts, 1):
        print(f"[{i}] {fragment}\n")
    return matched_texts

# 6. Llamar a Mistral local
def ask_local_llm(query):
    relevant_text = search_relevant_text(query)
    context = "\n".join(relevant_text)
    prompt = f"Texto de la ley:\n{context}\n\nPregunta: {query}\nRespuesta:"

    response = requests.post("http://localhost:5000/v1/chat/completions", json={
        "model": "mythomax-l2-13b",  # Usa el nombre del modelo que tu servidor reconoce
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    })

    print("🔄 STATUS CODE:", response.status_code)
    print("📦 RAW RESPONSE:", response.text)

    data = response.json()
    return data["choices"][0]["message"]["content"]

# 7. Hacer una pregunta
question = "¿En que fecha se publico la ley?"
answer = ask_local_llm(question)
print("\n🤖 Respuesta del modelo:\n")
print(answer)