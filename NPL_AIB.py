import os
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# Configura tu clave API
openai.api_key = 'Ingresa tu apikey'
# Función para cargar el archivo txt
def leer_archivo_txt(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        texto = archivo.read()
    return texto

# Función para calcular la similitud usando TF-IDF y Cosine Distances
def calcular_similitud(texto, pregunta):
    vector = TfidfVectorizer()
    matriz = vector.fit_transform([texto, pregunta])
    similitud = cosine_distances(matriz[0:1], matriz[1:2])
    return similitud[0][0]

# Función para realizar la pregunta a la API de OpenAI (limitada al contenido del archivo)
def realizar_pregunta_a_openai(pregunta, contenido_txt):
    prompt = (
        f"Tienes el siguiente archivo de texto:\n\n{contenido_txt}\n\n"
        "Responde la siguiente pregunta **únicamente** usando la información del archivo de texto anterior. "
        "Si la información no está presente en el archivo, responde con 'No se encuentra en el archivo'.\n\n"
        f"Pregunta: {pregunta}\n"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Puedes usar "gpt-4" si tienes acceso
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['choices'][0]['message']['content'].strip()

# Ruta del archivo de texto
ruta_archivo_ = 'BaseDeConocimentos.txt'

# Leer el contenido del archivo de texto
texto = leer_archivo_txt(ruta_archivo=ruta_archivo_)

# Input del usuario
pregunta = input("¿Qué deseas preguntar? ")

# Calcular la similitud (TF-IDF + Cosine Distance)
similitud = calcular_similitud(texto, pregunta)

# Mostrar la similitud
print(f"Similitud (TF-IDF + Cosine Distance): {similitud}")

# Realizar la pregunta a OpenAI
respuesta_openai = realizar_pregunta_a_openai(pregunta, texto)

# Mostrar la respuesta de OpenAI
print(f"Respuesta de OpenAI: {respuesta_openai}")
