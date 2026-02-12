import fitz  # PyMuPDF para leer el PDF
import requests

# 1. Extraer TODO el texto del PDF sin dividirlo
def extract_all_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    return full_text.strip()

# 2. Enviar el texto completo al modelo local (sin RAG, directo)
def ask_whole_pdf(query):
    context = extract_all_text("ley.pdf")
    prompt = f"""Analiza el siguiente texto de una ley chilena. Responde solo si encuentras información directamente en el texto, sin inventar.

Texto completo de la ley:
{context}

Pregunta: {query}
Respuesta:"""

    response = requests.post("http://localhost:5000/v1/chat/completions", json={
        "model": "Mistral-7B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    })

    print("🔄 STATUS CODE:", response.status_code)
    print("📦 RAW RESPONSE:", response.text)

    data = response.json()
    return data["choices"][0]["message"]["content"]

# 3. Hacer una pregunta
question = "¿Cuál es la fecha de publicación de esta ley?"
answer = ask_whole_pdf(question)
print("\n🤖 Respuesta del modelo:\n")
print(answer)
