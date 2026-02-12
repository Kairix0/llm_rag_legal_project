import fitz  # PyMuPDF para extraer texto de PDFs
from sentence_transformers import SentenceTransformer
import faiss
import openai


# 1. Cargar PDF y extraer texto
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    return text.split("\n")  # Divide en líneas

pdf_texts = extract_text_from_pdf("ley.pdf")

# 2. Convertir texto en embeddings (usamos un modelo pequeño)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(pdf_texts)

# 3. Crear FAISS index para búsqueda rápida
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 4. Buscar el fragmento más relevante según la pregunta
def search_relevant_text(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [pdf_texts[i] for i in indices[0]]

# 5. Enviar contexto al LLM para generar respuesta
def ask_llm(query):
    relevant_text = search_relevant_text(query)
    context = "\n".join(relevant_text)
    prompt = f"Texto de la ley:\n{context}\n\nPregunta: {query}\nRespuesta:"
    
    client = openai.OpenAI()  # Se crea el cliente

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Eres un asistente legal."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# 6. Hacer una pregunta
question = "¿Cuál es la clasificación de las películas según esta ley?"
answer = ask_llm(question)
print(answer)