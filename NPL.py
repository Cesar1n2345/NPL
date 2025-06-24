import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


# Determinar una funcion para cargar el archivo txt

def leer_archivo_txt(ruta_archivo):
    with open(ruta_archivo,'r', encoding='utf-8') as archivo:
        texto = archivo.read()
    return texto

def calcular_similitud(texto,pregunta):
    vector = TfidfVectorizer()

    matriz = vector.fit_transform([texto,pregunta])

    similitud = cosine_distances(matriz[0:1], matriz[1:3])

    return similitud[0][0]

ruta_archivo_ = 'BaseDeConocimentos.txt'

texto = leer_archivo_txt(ruta_archivo=ruta_archivo_)


vectorizar = TfidfVectorizer()

matriz_vectorizada = vectorizar.fit_transform([texto])


nombres_estructuras = vectorizar.get_feature_names_out()
puntaje_comparacion = matriz_vectorizada.toarray()

pregunta = input("Que deseas preguntar")

mostrar_similitud = calcular_similitud(texto,pregunta)

print(mostrar_similitud)

"""for nombres,puntaje in zip(nombres_estructuras, matriz_vectorizada[0]):
    print(f"Nombres = {nombres} y el puntaje = {puntaje}")
"""




