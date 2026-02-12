import fitz  # PyMuPDF para extraer texto de PDFs
from sentence_transformers import SentenceTransformer
import faiss
import requests
import torch
#Librerias agregadas para la creacion del data set
import json
import uuid
#Librerias agregadas para la implementacion de memoria de contexto
import pickle
import os

assert torch.cuda.is_available(), "❌ La GPU no está disponible. Asegúrate de tener los drivers y CUDA correctamente instalados."

# 📄 1. Extraer texto del PDF
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

# 📦 Cargar y procesar texto del PDF solo una vez
pdf_path = "Leyes_para_dataSets/Ley_de_viviendas/Ley_19281.pdf"
pdf_texts = extract_text_from_pdf(pdf_path)

# 🧠 2. Modelo y GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("intfloat/multilingual-e5-base")
embeddings = model.encode(pdf_texts, device=device, show_progress_bar=True, batch_size=32)

# Mover explícitamente el modelo a CUDA
model.to(torch.device(device))

print("📡 Dispositivo actual:", device)
if device == "cuda":
    print("🚀 GPU detectada:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU no detectada o no disponible. Usando CPU.")

# 🧱 3. Crear o reutilizar índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 🔍 4. Buscar texto relevante
def search_relevant_text(query, top_k=4, max_total_chars=1500):
    query_embedding = model.encode([query], device=device)
    distances, indices = index.search(query_embedding, top_k)
    matched_texts = [pdf_texts[i] for i in indices[0]]
    total_charts = 0
    for i in indices[0]:
        chunk = pdf_texts[i]
        if total_charts + len(chunk) <= max_total_chars:
            matched_texts.append(chunk)
            total_charts += len(chunk)
    return matched_texts

# 🤖 5. LLM local
def ask_local_llm(query):
    relevant_text = search_relevant_text(query)
    context = "\n".join(relevant_text)
    prompt = f"Texto de la ley:\n{context}\n\nPregunta: {query}\nRespuesta:"

    response = requests.post("http://localhost:5000/v1/chat/completions", json={
        "model": "Mistral-7B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens":300,
        "stop": ["\n\n"],
    })

    data = response.json()
    return data["choices"][0]["message"]["content"]

# 💾 6. Funciones de caché
def load_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# 🧰 7. Dataset builder
def build_dataset(questions, dataset_path, cache_path="Leyes_para_dataSets/Ley_de_viviendas/qa_cache_ley_viviendas.json"):
    dataset = []
    qa_cache = load_cache(cache_path)

    # 📂 Cargar dataset anterior si existe
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    
    existing_questions = set(entry["input"] for entry in dataset)

    for question in questions:
        print(f"\n🔎 Pregunta: {question}")

        if question in existing_questions:
            print("⚠️ Pregunta ya existe en el dataset. Saltando...")
            continue

        # Usar respuesta cacheada si existe
        if question in qa_cache:
            print("✅ Usando respuesta desde caché.")
            answer = qa_cache[question]
        else:
            answer = ask_local_llm(question).strip()
            qa_cache[question] = answer
            save_cache(qa_cache, cache_path)

        # Agregar nueva entrada
        entry = {
            "id": str(uuid.uuid4()),
            "input": question,
            "output": answer,
            "expected_keyword": []
        }
        dataset.append(entry)

    # Guardar dataset final
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Dataset guardado en {dataset_path}")

# 🧪 8. Preguntas de prueba
questions = [
    "¿Por que no se puede exceder el veinticinco por ciento de la renta liquida mensual que acredite tener al momento de celebrar el contrato de arrendamiento con promesa de compraventa?",
"¿En que se basara la renta liquida?",
"¿Que retribucion obtendran las instituciones segun esta ley?",
"¿La comision sera establecida libremente por cada institucion?",
"¿Es necesario que el interesado este suscrito al contrato de arrendamiento de la vivienda?"
]

# 9. Contador de preguntas registradas en el archivo .JSON
def count_questions_in_dataset(json_path):
    if not os.path.exists(json_path):
        print("❌ El archivo no existe.")
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        total = len(data)
        print(f"📊 Total de preguntas registradas: {total}")
        return total

build_dataset(questions, dataset_path="Leyes_para_dataSets/Ley_de_viviendas/dataset_ley_viviendas.json")
count_questions_in_dataset("Leyes_para_dataSets/Ley_de_viviendas/dataset_ley_viviendas.json")