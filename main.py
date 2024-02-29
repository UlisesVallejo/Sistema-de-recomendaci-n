from flask import Flask, request, jsonify
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

spacy.cli.download("es_core_news_sm")
nlp = spacy.load('es_core_news_sm')

#Función para normalizar la data
def normalizar(texto):
    #convertir a minúsculas
    lower = texto.lower()
    # remover numeros
    no_numbers = re.sub(r'\d', '', lower)
    #eliminar puntuaciones
    no_punt = re.sub(r'[^\w\s]', '', no_numbers)
    #eliminar espacios
    no_space = no_punt.strip()
    ##############
    array_words = [no_space][0].split()
    
    stop_words = set(stopwords.words('spanish'))
    
    #eliminar stop_words
    no_stopwords_string=""
    for palabra in array_words:
        if not palabra in stop_words:
            no_stopwords_string += palabra+' '
            
    no_stopwords_string = no_stopwords_string[:-1]
    return no_stopwords_string.split()


app = Flask(__name__)

@app.route('/recomendaciones', methods=['POST'])
def recomendar():
    
    data = request.json
    
    perfil = data.get("perfil")
    plazas = data.get("plazas")
    
    resultado = normalizar(perfil.get("actividades"))   #RECIBE TEXTO DEL DATA

    lemas = [token.lemma_ for token in nlp(" ".join(resultado))]
    
    resultado_carr = normalizar(perfil.get("carrera"))

    lema_carrera = [token.lemma_ for token in nlp(" ".join(resultado_carr).upper())]

    perfil_usuario = {"Carrera": perfil.get("carrera"), "Actividades": lemas}

    dependencias = []

    for dependencia in plazas:
        dependencias.append({
            "nombre": dependencia.get("nombre"),
            "carrera": dependencia.get("carrera"),
            "actividades": dependencia.get("actividades"),
            "actividades_lema": [token.lemma_ for token in nlp(" ".join(normalizar(dependencia.get("actividades"))))]
        })    
    

    # Sacamos una lista con las actividades
    newact = []
    for elem in dependencias:
        for k,v in elem.items():        #acedemos a cada llave(k), valor(v) de cada diccionario
            if k == "actividades_lema":
                for act in v:
                    newact.append(act)
                    

    # Eliminar duplicados de las actividades de usuario y de dependencias
    lemas_unicos_actividades_usuario = []
    for i in lemas:
        if i not in lemas_unicos_actividades_usuario:
            lemas_unicos_actividades_usuario.append(i)
            
    lemas_unicos_actividades_dep = []
    for i in newact:
        if i not in lemas_unicos_actividades_dep:
            lemas_unicos_actividades_dep.append(i)


    todas_actividades = lemas_unicos_actividades_usuario + lemas_unicos_actividades_dep #unimmos las listas

    # Asignar valores numéricos a cada elemento de la lista
    actividades_numericos = list(range(1, len(todas_actividades) + 1))
    actividades_numericos2 = list(range(1, len(lemas_unicos_actividades_dep) + 1))

    # Crear un diccionario que asocie cada palabra con su valor numérico
    diccionario_actividades = dict(zip(todas_actividades, actividades_numericos))
    diccionario_actividades_onlydep = dict(zip(lemas_unicos_actividades_dep, actividades_numericos2))

    # Imprimir el resultado
    #print(diccionario_actividades_onlydep)


    # Sacamos una lista con las carreras
    newcarr = []
    for elem in dependencias:
        for k,v in elem.items():        #acedemos a cada llave(k), valor(v) de cada diccionario
            if k == "carrera":
                for carr in v:
                    newcarr.append(carr)
                    
    # Eliminar duplicados carreras
    carreras_unicas_usuario = []
    for i in lema_carrera:
        if i not in carreras_unicas_usuario:
            carreras_unicas_usuario.append(i)

    carreas_unicas_dependencias = []
    for i in newcarr:
        if i not in carreas_unicas_dependencias:
            carreas_unicas_dependencias.append(i)

    todas_carreras = carreras_unicas_usuario + carreas_unicas_dependencias 

    carreras_numericos = list(range(1, len(todas_carreras) + 1))
    diccionario_carreras = dict(zip(todas_carreras, carreras_numericos))
    print(diccionario_carreras)

    # Representación numérica del perfil del usuario
    perfil_numerico = np.zeros(len(diccionario_carreras) + len(diccionario_actividades))
    perfil_numerico[diccionario_carreras[perfil.get("carrera")]] = 1
    for actividad in perfil_usuario["Actividades"]:
        perfil_numerico[len(diccionario_carreras) + diccionario_actividades[actividad]] = 1
        
        
    # Representación numérica de las dependencias gubernamentales
    dependencias_numericas = []
    for dep in dependencias:
        dep_numerica = np.zeros(len(diccionario_carreras) + len(diccionario_actividades))
        #print(dep_numerica)
        for carrera in dep["carrera"]:#Esta agarrando las carreras de la dependencia
            dep_numerica[diccionario_carreras[carrera]] = 1 #Le estoy poniendo todas las carreras
        for actividad in dep["actividades_lema"]:
            try:
                dep_numerica[len(diccionario_carreras) + diccionario_actividades[actividad]] = 1
            except:
                pass
            
        dependencias_numericas.append(dep_numerica)
        

    # Calcula la similitud coseno entre el perfil del usuario y las dependencias gubernamentales
    similitudes = [cosine_similarity([perfil_numerico], [dep_numerica])[0][0] for dep_numerica in dependencias_numericas]

    # Ordena las dependencias por similitud descendente
    resultados = sorted(zip(dependencias, similitudes), key=lambda x: x[1], reverse=True)
    
    
    return jsonify({'recomendaciones': resultados})

if __name__ == '__main__':
    app.run(debug=True)