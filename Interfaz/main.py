import os
import sys
import shutil
import urllib.request
import joblib
# imports de la interfaz
from PyQt5 import QtCore, QtWidgets, QtGui, uic
from PyQt5.QtCore import QSize
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import webbrowser
import glob
import re
import requests
import logging
import json
import multiprocessing
import subprocess
import time


def install(package):
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", package])


try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
except ModuleNotFoundError:
    install("selenium")
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys

try:
    from webdriver_manager.chrome import ChromeDriverManager
except ModuleNotFoundError:
    install("webdriver_manager")
    from webdriver_manager.chrome import ChromeDriverManager
try:
    import pandas as pd
except ModuleNotFoundError:
    install("pandas")
    import pandas as pd
try:
    from pytube import YouTube
    from pytube import Playlist
except ModuleNotFoundError:
    install("pytube")
    from pytube import YouTube
    from pytube import Playlist
try:
    import speech_recognition as sr
except ModuleNotFoundError:
    install("SpeechRecognition")
    import speech_recognition as sr
try:
    from pydub import AudioSegment
except:
    install("pydub")
    from pydub import AudioSegment
try:
    import moviepy.editor as mp
except:
    install("moviepy")
    import moviepy.editor as mp
try:
    from bs4 import BeautifulSoup
except:
    install("beautifulsoup4")
    from bs4 import BeautifulSoup
try:
    from nltk.tokenize import word_tokenize
except:
    install("nltk")
    from nltk.tokenize import word_tokenize
try:
    import pyrebase
except:
    install("pyrebase4")
    import pyrebase
try:
    import nltk
    nltk.download('stopwords')
    from nltk.stem.rslp import RSLPStemmer
    nltk.download('rslp')
except:
    install("nltk")
    import nltk
    nltk.download('stopwords')
    from nltk.stem.rslp import RSLPStemmer
    nltk.download('rslp')
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import ParameterGrid
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
except ModuleNotFoundError:
    install("scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import ParameterGrid
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
try:
    import numpy as np
except ModuleNotFoundError:
    install("numpy")
    import numpy as np
try:
    from keras.models import Sequential
    from keras import layers
except ModuleNotFoundError:
    install("keras")
    from keras.models import Sequential
    from keras import layers


class ControladorVideo:
    def __init__(self, enlace):
        fb = Firebase('recetastextos/')
        self._idvideo = fb.reenumerar()
        self.enlacevideo = enlace
        self.yt = YouTube(self.enlacevideo)
        self.nombrevideo = ''
        self.titulovideo = self.yt.title
        self.autorvideo = self.yt.author
        self.fechavideo = self.yt.publish_date
        self.duracionvideo = self.yt.length
        self.rec = RecursosAdicionales()
    """|DESCARGAR VIDEO URL: descarga el video de youtube
       |return: devuelve una ruta absoluta"""

    def descargarVideoURL(self):
        # aqui creo un nuevo id para el nuevo video
        self._idvideo = self._idvideo+1
        # le pedimos al pytube que solo nos descargue el audio y lo descargamos
        self.yt.streams.filter(file_extension='mp4').first().download(
            output_path='recetasvideos/', filename='receta'+str(self._idvideo)+'.mp4')
     
    """|PARSEO VIDEO: pasa el video de .mp4 a .wav
       |nombre: es un string que se colocara el nombre del video
       |return: devuelve el nuevo nombre del audio en .wav"""
    def parseoVideo(self, nombre):
        recetasVideos = 'recetasvideos/'
        # tomamos el video en mp4
        track = mp.VideoFileClip(recetasVideos+nombre+'.mp4')
        # cambiamos el video a .wav
        nombre_wav = "{}.wav".format(nombre)
        track.audio.write_audiofile(recetasVideos+nombre_wav)
        track.close()
        return nombre
    """|SPEECH TEXT:Transforma el audio a texto
       |nombre: es un string que se colocara el nombre del video
       |return: devuelve un string con el texto devuelto"""

    def speech_text(self, nombre):
        recetasVideos = 'recetasvideos/'
        # instanciamos el recognizer
        r = sr.Recognizer()
        audio = sr.AudioFile(recetasVideos+nombre)
        with audio as source:
            audio_file = r.record(source)
        # transcribimos el audio a texto
        result = r.recognize_google(audio_file, language='es-ES')
        return result

    def data_json(self):
        return {"id": self._idvideo, "nombre": self.titulovideo, "autor": self.autorvideo, "fecha": str(self.fechavideo), "enlace": str(self.enlacevideo)}

    def indexar_datos(self):
        return self.rec.indexar_datos("recetastextos/indice.json", {"id": self._idvideo+1, "nombre": self.titulovideo, "autor": self.autorvideo, "fecha": str(self.fechavideo), "enlace": str(self.enlacevideo)})
    """|REPETIDO:Nos dice si el video ya se encuentra en nuestra bd
       |fileName: nombre del json
       |key: llave en donde queremos encontrar lo que buscamos
       |buscar: elemento que estamos buscando"""

    def repetido(self):
        return self.rec.buscar_json('recetastextos/indice.json', 'nombre', self.titulovideo)
    def esLista(self):
        if(re.search("/playlist?", self.enlacevideo)):
            return True
        else:
            return False
    def video(self):
        try:
            # instanciamos el controlador de videos
            fb = Firebase('recetastextos/')

            # paso 1: verificamos si existe en la database
            if fb.validar_database(self.titulovideo) == False:
                # paso 2: guardamos en database datos principales

                # paso 3: descargamos el video
                self.nombrevideo = self.descargarVideoURL()
                print("id: "+str(self._idvideo))
                fb.guardar_database(self.data_json(), self._idvideo)
                # paso 4: pasamos el video a .wav
                nombre = self.parseoVideo(self.nombrevideo)
                # paso 5: evaluamos los silencios
                try:
                    num_segm = self.rec.segcionarXsilencios(nombre)
                    result = ""
                    for i in range(num_segm):
                        try:
                            result = result + \
                                str(self.speech_text(
                                    "../temp_audios/{}_extracto{}.wav".format(nombre, i+1)))
                            result = result+" "
                        except BaseException:
                            logging.exception("An exception was thrown!")
                            audio1 = AudioSegment.from_wav(
                                "temp_audios/{}_extracto{}.wav".format(nombre, i+1))
                            duracion = audio1.duration_seconds
                            if duracion <= 5:
                                print("El extracto {} es un silencio".format(i+1))
                            elif duracion <= 180:
                                print(
                                    "El extracto {} es música o ruido".format(i+1))
                            else:
                                print(
                                    "Error importante en el extracto {}".format(i+1))
                    # paso 6: borramos los chunks temporales de audio
                    self.rec.eliminacion_audio("temp_audios", "wav")
                    try:
                        quitarEmojis = dict.fromkeys(
                            range(0x10000, sys.maxunicode + 1), 'NULL')
                        tituloSinEmojis = self.titulovideo.translate(
                            quitarEmojis)
                        autorSinEmojis = self.autorvideo.translate(
                            quitarEmojis)
                        # paso 7: escribimos el texto recibido en un txt->se guarda en local
                        resultado = self.rec.escritura(self.nombrevideo, "Titulo:"+tituloSinEmojis+"\n"+"Autor:"+autorSinEmojis+"\n"+"Fecha Publicacion:"+str(
                            self.fechavideo)+"\n"+"Enlace: "+str(self.enlacevideo)+"\n"+"Entradilla:"+result)
                        # paso 8: guardamos el texto en una base de datos
                        fb.guardar_firebase(self.nombrevideo+'.txt')
                        # paso 9: eliminamos los mp4
                        self.rec.eliminacion_audio("recetasvideos", "mp4")
                    except BaseException:
                        logging.exception("An exception was thrown!")
                        print("No se ha podido eliminar los caracteres corruptos el video: " +
                              self.nombrevideo + " - " + self.titulovideo)
                        self.rec.eliminacion_audio("recetasvideos", "mp4")
                        return None
                except BaseException:
                    logging.exception("An exception was thrown!")
                    print("No se ha podido transcribir el video: " + self.nombrevideo +
                          " - " + self.titulovideo+" - "+self.enlacevideo)
                    self.rec.eliminacion_audio("recetasvideos", "mp4")
                    self.rec.eliminacion_audio("temp_audios", "wav")
                    return None
            else:
                print('Este video se encuentra en la base de datos.')
                resultado = ""
            return resultado
        except BaseException:
            logging.exception("An exception was thrown!")
            print("No se ha podido descargar el video: " +
                  self.nombrevideo + " - " + self.titulovideo)
            return None
    def lista(self):
        playlist_urls = Playlist(self.enlacevideo)
        for url in playlist_urls:
            self.video(url)



    
    


class RecursosAdicionales:
    """|ESCRITURA: escribe textos txt
       |nombre: nombre del 
       |return: devuelve el audio en texto"""

    def escritura(self, nombre, texto):
        recetasTextos = 'recetastextos/'
        if not(os.path.exists(recetasTextos)):
            os.mkdir(recetasTextos)
        f = open(recetasTextos+nombre+'.txt', 'w')
        f.write(texto)
        f = open(recetasTextos+nombre+'.txt', "r")
        print(f.read())
        f.close()

    def lectura_json(self, fileName):
        if self.documento_vacio(fileName):
            with open(fileName, "r") as file:
                archivo = json.load(file)
        else:
            archivo = []
            print('El documento se encuentra vacio.')
        return archivo

    def escritura_json(self, fileName, data):
        with open(fileName, "w") as file:
            json.dump(data, file)
            file.close()

    def buscar_json(self, fileName, key, buscar):
        encontrado = False
        if self.documento_vacio(fileName):
            archivo_json = self.lectura_json(fileName)
            for item in archivo_json:
                if buscar in item[key]:
                    print('encontrado')
                    encontrado = True
                    # no me gusta usar esto pero no tengo idea de como usar un while con json
                    break
        return encontrado

    def documento_vacio(self, fileName):
        return os.stat(fileName).st_size != 0

    def indexar_datos(self, fileName, adicion):
        if not(os.path.exists(fileName)):
            os.mkdir(fileName)
        data = []
        data = self.lectura_json(fileName)
        data.append(adicion)
        self.escritura_json(fileName, data)

    def eliminacion_audio(self, path, tipo):
        url = './'+path+'/'
        py_files = glob.glob(url+'*.'+tipo)
        for py_file in py_files:
            try:
                os.remove(py_file)
            except OSError as e:
                print(f"Error:{ e.strerror}")

    def segcionarXsilencios(self, audio):
        audio1 = AudioSegment.from_wav("./recetasvideos/"+audio+".wav")
        var_min = 1900
        salir = False
        while salir == False:
            samples = audio1.get_array_of_samples()
            segundo = 88521
            index = []
            for i in range(0, len(samples), int(segundo/5)):
                dataSeg = samples[i:int(segundo/5)+i]
                media = np.mean(dataSeg)
                var = np.var(dataSeg)
                if -10 <= media <= 10 and var <= var_min:
                    index.append(i)

            borrar = []
            guardado = 0
            for i in range(len(index)-1):
                if index[i+1] <= index[i]+(20*segundo):
                    if i == 0:
                        tiempo = (index[i])/segundo
                    else:
                        tiempo = (index[i+1]-guardado)/segundo
                    if tiempo <= 120:
                        borrar.append(i)
                    else:
                        guardado = index[i]
                else:
                    guardado = index[i]

            final = np.delete(index, borrar, axis=0)
            extractos = []
            if len(final) == 0:
                var_min = var_min*10
                salir = False
            else:
                for i in range(len(final)):
                    if i == 0:
                        extractos.append(samples[:final[i]])
                    else:
                        extractos.append(samples[final[i-1]:final[i]])
                extractos.append(samples[final[i]:])
                salir = True

        for i in range(len(extractos)):
            nombre = ""
            new_sound = audio1._spawn(extractos[i])
            nombre = "temp_audios/{}_extracto{}.wav".format(audio, i+1)
            new_sound.export(nombre, format="wav")
        # print(len(extractos))
        return len(extractos)


class Firebase:
    def __init__(self, ubicacion):
        self.ubi = ubicacion

        self.config = {"apiKey": "AIzaSyDDg9WOlFJxnEJoxomYtsnkJfsI4TgoL_E", "authDomain": "eateaser-741d4.firebaseapp.com", "databaseURL": "https://eateaser-741d4-default-rtdb.firebaseio.com/",
                       "projectId": "eateaser-741d4", "storageBucket": "eateaser-741d4.appspot.com", "messagingSenderId": "706351391410", "appId": "1:706351391410:web:6abc2cabd6bf83843b5fab", "measurementId": "G-YZZCBRHNBT"}
        self.firebase = self.conexion_firebase()
        self.database = self.firebase.database()

    def conexion_firebase(self):
        return pyrebase.initialize_app(self.config)

    def guardar_firebase(self, nom):
        storage = self.firebase.storage()
        storage.child(self.ubi+nom).put(self.ubi+nom)

    def eliminar_firebase(self, nom):
        self.firebase.storage().delete(self.ubi+nom)

    def guardar_database(self, data, _id):
        self.database.child('Recetas').child(_id).set(data)

    def validar_database(self, data):
        validar = self.database.get()
        encontrado = False
        for a in validar.each():
            if data in str(a.val()):
                encontrado = True
                # no me gusta usar esto pero no tengo idea de como usar un while con json
                break
        return encontrado

    def reenumerar(self):
        recetas = self.database.child("Recetas").get()
        id = 0
        for item in recetas.each():
            id = item.key()
        return int(id)


class ProcesarDocumentos:
    def lectura(self):
        procDoc = ProcesarDocumentos()
        rutaCarpetasPorCategoria = "./recetastextos/"
        listaCarpetasFinal = []
        # estos string nos servirán para guardar todos los textos de los txt por cada una de las carpetas
        carpetaArroz = carpetaBebidas = carpetaCarnes = carpetaMarisco = carpetaPasta = carpetaPescados = carpetaPlatosMenores = carpetaVerduras = ''
        # sacamos una lista de todas las carpetas
        listaCarpetas = os.listdir(rutaCarpetasPorCategoria)
        # print(listaCarpetas)
        # print(len(listaCarpetas))
        # recorremos todas las carpetas
        i = 0
        for lc in listaCarpetas:
            # cogemos el nombre de la carpeta y se lo concatenamos a la ruta anterior
            rutaPorCarpeta = rutaCarpetasPorCategoria + lc + '/'
            print(str(i)+rutaPorCarpeta+'-----------------')
            if(i == 0):
                carpetaArroz = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                # print(carpetaArroz)
                listaCarpetasFinal.append(carpetaArroz)
            if(i == 1):
                carpetaBebidas = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaBebidas)
            if(i == 2):
                carpetaCarnes = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaCarnes)
            if(i == 3):
                carpetaMarisco = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaMarisco)
            if(i == 4):
                carpetaPasta = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaPasta)
            if(i == 5):
                carpetaPescados = procDoc.resultadoStringCarpeta(
                    rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaPescados)
            if(i == 6):
                carpetaPlatosMenores = procDoc.resultadoStringCarpeta(
                    rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaPlatosMenores)
            if(i == 7):
                carpetaVerduras = procDoc.resultadoStringCarpeta(
                    rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaVerduras)
            i = i+1
        return listaCarpetasFinal

    def lecturaTesting(self):
        procDoc = ProcesarDocumentos()
        rutaCarpetaTesting = "./Interfaz/Carpeta Testing/"
        carpetaTesting = procDoc.resultadoStringCarpeta(rutaCarpetaTesting)
        return carpetaTesting

    def resultadoStringCarpeta(self, rutaPorCarpeta):
        strCarpeta = []
        # vemos el contenido de la carpeta en la que estamos iterando
        listaTxt = os.listdir(rutaPorCarpeta)
        print(listaTxt)
        # recorremos todos los archivos de la carpeta
        for lt in listaTxt:
            # concatenamos la ruta de la carpeta con el nombre de los archivos que contiene esta
            rutaTxt = rutaPorCarpeta + lt
            # al ir iterando pasaremos por todos los archivos modificando la variable de la ruta para poder hacer un open con ella
            #file = open(filename, encoding="utf8")
            try:
                with open(rutaTxt, 'r') as f:
                    # al hacer el open leemos lo que hay dentro del archivo con f.read(), y esto lo guardamos dentro de un string inicializado al inicio del todo
                    strCarpeta.append(f.read())
            except:
                with open(rutaTxt, 'r', encoding="utf8") as f:
                    # al hacer el open leemos lo que hay dentro del archivo con f.read(), y esto lo guardamos dentro de un string inicializado al inicio del todo
                    strCarpeta.append(f.read())

        return strCarpeta

    def leer_stopwords(self, path):
        with open(path) as f:
            # Lee las stopwords del archivo y las guarda en una lista
            mis_stopwords = [line.strip() for line in f]
        return mis_stopwords

    def tratamientoTextos(self, info):
        # Eliminamos posibles horas del titulo
        textoSinSimbolos = re.sub("\d+:\d+:\d+", "", info)
        # Eliminamos posibles fechas
        textoSinSimbolos = re.sub("\d+-\d+-\d+", "", textoSinSimbolos)
        # Eliminamos todos los fin de enlace
        textoSinSimbolos = re.sub("v=.*", "", textoSinSimbolos)
        # Eliminamos todos los simbolos del texto (,.;:?¿!!) etc
        textoSinSimbolos = re.sub("[^0-9A-Za-z_]", " ", textoSinSimbolos)
        # Sacamos todos los tokens del texto y los metemos en una lista
        textoTokenizado = nltk.tokenize.word_tokenize(textoSinSimbolos)
        # una lista no tiene lower asique pasamos el lower con map a toda la lista
        textoMinusculas = (map(lambda x: x.lower(), textoTokenizado))
        # Le pasa un stopword de palabras en español a la lista de palabras que le llega
        #stop_words_sp = set(stopwords.words('spanish'))
        stop_words_sp = self.leer_stopwords(
            "./rapidminer/stop_words_spanish.txt")
        pasarStopWords = [i for i in textoMinusculas if i not in stop_words_sp]
        # Aplicamos la normalizacion mediante stemming
        #SnowStem = nltk.SnowballStemmer(language = 'spanish')
        # Crear un objeto SnowballStemmer para el idioma español
        stemmer = RSLPStemmer()
        listaStems = [stemmer.stem(word) for word in pasarStopWords]
        return listaStems

    def Entrenar_SVM(self):
        # self.preprocesamiento()

        # Create a svm Classifier
        clf = svm.SVC(kernel='linear')  # Linear Kernel

        # Train the model using the training sets
        clf.fit(self.X_train, self.Y_train)

        predictions = clf.predict(self.X_cv)
        self.Y_cv = list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))

        cv = cross_val_score(clf, self.X_train, self.Y_train, cv=10)

        print("CV -> {}".format(cv))

    def Entrenar_SVM_CV(self):

        Cs = np.logspace(-6, -1, 10)
        svc = svm.SVC()
        clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),
                           n_jobs=-1)
        clf.fit(self.X_train, self.Y_train)

        print("best score-> {}".format(clf.best_score_))

        print("best estimator-> {}".format(clf.best_estimator_.C))

        # Prediction performance on test set is not as good as on train set
        print("mi score".format(clf.score(self.X_cv, self.Y_cv)))


class modelosTFIDF:
    def __init__(self, df, features_mod):
        self.df = df
        self.tfidf(features_mod)

    def tfidf(self, features_mod):
        hola = []
        for i, j in enumerate(self.df['receta']):
            hola.append(" ".join(j))
        self.vectorizers = TfidfVectorizer(max_features=features_mod)
        self.vect = self.vectorizers.fit_transform(hola)
        arr = self.vect.toarray()
        variable = self.vectorizers.get_feature_names()

        variables = dict.fromkeys(variable, None)

        tf1 = pd.DataFrame(variables, index=[0])
        for i in range(len(self.df['clasif'])):
            tf1.loc[i] = arr[i]

        self.X_train, self.X_cv, self.Y_train, self.Y_cv = train_test_split(
            tf1, self.df['clasif'], test_size=0.2, random_state=42)
        self.Y_train = list(self.Y_train)
        self.Y_cv = list(self.Y_cv)

    def Entrenar_RF(self):
        # Grid de hiperparámetros evaluados
        # ==============================================================================

        print(self.X_train.shape)
        param_grid = ParameterGrid(
            {'n_estimators': [1000],
             'max_features': [5, 7, 9],
             'max_depth': [None, 3, 10, 20],
             'criterion': ['gini', 'entropy']
             }
        )

        # Loop para ajustar un modelo con cada combinación de hiperparámetros
        # ==============================================================================
        resultados = {'params': [], 'oob_accuracy': []}

        for params in param_grid:

            modelo = RandomForestClassifier(
                oob_score=True,
                n_jobs=-1,
                random_state=123,
                ** params
            )

            modelo.fit(self.X_train, self.Y_train)

            resultados['params'].append(params)
            resultados['oob_accuracy'].append(modelo.oob_score_)
            print(f"Modelo: {params} \u2713")

        # Resultados
        # ==============================================================================
        resultados = pd.DataFrame(resultados)
        resultados = pd.concat(
            [resultados, resultados['params'].apply(pd.Series)], axis=1)
        resultados = resultados.sort_values('oob_accuracy', ascending=False)
        resultados = resultados.drop(columns='params')
        print(resultados.head(4))

        '''
        self.forest = RandomForestClassifier() 
        self.forest = self.forest.fit(self.X_train, self.Y_train)

        predictions = self.forest.predict(self.X_cv) 
        self.Y_cv=list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))
        '''

    def Entrenar_RF_CV(self):
        # Grid de hiperparámetros evaluados
        # ==============================================================================
        param_grid = {'n_estimators': [150],
                      'max_features': [5, 7, 9],
                      'max_depth': [None, 3, 10, 20],
                      'criterion': ['gini', 'entropy']
                      }

        # Búsqueda por grid search con validación cruzada
        # ==============================================================================

        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=123),
            param_grid=param_grid,
            scoring='accuracy',
            n_jobs=multiprocessing.cpu_count() - 1,
            cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
            refit=True,
            verbose=0,
            return_train_score=True
        )

        grid.fit(X=self.X_train, y=self.Y_train)

        # Resultados
        # ==============================================================================
        resultados = pd.DataFrame(grid.cv_results_)
        resultados.filter(regex='(param*|mean_t|std_t)') \
            .drop(columns='params') \
            .sort_values('mean_test_score', ascending=False) \
            .head(4)

        # Mejores hiperparámetros por validación cruzada
        # ==============================================================================
        print("----------------------------------------")
        print("Mejores hiperparámetros encontrados (cv)")
        print("----------------------------------------")
        print(grid.best_params_, ":", grid.best_score_, grid.scoring)

        self.modelo_final = grid.best_estimator_

        '''
        self.forest = RandomForestClassifier() 
        self.forest = self.forest.fit(self.X_train, self.Y_train)

        predictions = self.forest.predict(self.X_cv) 
        self.Y_cv=list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))
        '''

    def Entrenar_SVM(self):
        # self.preprocesamiento()

        # Create a svm Classifier
        m_SVM = svm.SVC(kernel='linear')  # Linear Kernel

        # Train the model using the training sets
        m_SVM.fit(self.X_train, self.Y_train)

        predictions = m_SVM.predict(self.X_cv)
        self.Y_cv = list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))

        cv = cross_val_score(m_SVM, self.X_train, self.Y_train, cv=10)
        self.m_SVM = m_SVM

        print("CV -> {}".format(cv))
        return self.m_SVM

    def Entrenar_SVM_CV(self):

        Cs = np.logspace(-6, -1, 10)
        svc = svm.SVC()
        m_SVM_CV = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),
                                n_jobs=-1)
        m_SVM_CV.fit(self.X_train, self.Y_train)

        print("best score-> {}".format(m_SVM_CV.best_score_))

        print("best estimator-> {}".format(m_SVM_CV.best_estimator_.C))

        # Prediction performance on test set is not as good as on train set
        print("mi score".format(m_SVM_CV.score(self.X_cv, self.Y_cv)))

    def Entrenar_Bayes(self):

        # Create a svm Classifier
        gaus = GaussianNB()  # Linear Kernel

        # Train the model using the training sets
        gaus.fit(self.X_train, self.Y_train)

        predictions = gaus.predict(self.X_cv)
        self.Y_cv = list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))

        cv = cross_val_score(gaus, self.X_train, self.Y_train, cv=10)
        self.gaus = gaus
        print("CV -> {}".format(cv))

    def Entrenar_RegresionMultinomial(self):

        M_mult = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        M_mult.fit(self.X_train, self.Y_train)

        predictions = M_mult.predict(self.X_cv)
        self.Y_cv = list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))

        #cv=cross_val_score(M_mult, self.X_train, self.Y_train, cv=10)
        self.M_mult = M_mult
        #print("CV -> {}".format(cv))
        return self.M_mult

    def predecir_RF(self, txt):

        self.pred = self.vectorizers.transform(txt)
        self.pred = self.pred.toarray()
        predictions = self.M_mult.predict(self.pred)
        print("resultado: ", predictions)

    def clasificar(self, modelo, txt):

        self.pred = self.vectorizers.transform(txt)
        self.pred = self.pred.toarray()
        predictions = modelo.predict(self.pred)
        print("resultado: ", predictions)

    def predecir_Carpeta(self, rutaModelo, modeloSeleccionado):

        p = ProcesarDocumentos()
        carpeta = p.resultadoStringCarpeta(rutaModelo)

        # resultados=[]
        # for i in range(len(carpeta)):
        #    text=p.tratamientoTextos(carpeta[i])
        #    hey=[" ".join(text)]
        #    resultados.append(self.predecir_RF(hey))
        #print("Resultados: {}".format(resultados))

        resultados = []
        for i in range(len(carpeta)):
            text = p.tratamientoTextos(carpeta[i])
            hey = " ".join(text)
            resultados.append(hey)
        self.pred1 = self.vectorizers.transform(resultados)
        self.pred1 = self.pred1.toarray()
        predictions = modeloSeleccionado.predict(self.pred1)
        #print("resultado: " , predictions)
        return predictions

    def Entrenar_KNN(self):
        # self.preprocesamiento()

        vecinos = KNeighborsClassifier()
        vecinos = vecinos.fit(self.X_train, self.Y_train)

        predictions = vecinos.predict(self.X_cv)
        self.Y_cv = list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))

    def Entrenar_RedNeuronal(self):
        xtrain = []
        for i in range(len(self.X_train)):
            xtrain.append(list(self.X_train.iloc[i]))
        # xtrain=np.array(xtrain)

        xcv = []
        for i in range(len(self.X_cv)):
            xcv.append(list(self.X_cv.iloc[i]))
        # xcv=np.array(xcv)
        # Y_train=np.array(Y_train)
        # Y_cv=np.array(Y_cv)

        print(type(xtrain))
        print(type(self.Y_train))
        print(type(xcv))
        print(type(self.Y_cv))

        clear_session()

        # input_dim = xtrain.shape[1] #.shape[0]  # Number of features
        input_dim = len(xtrain[0])
        model = Sequential()
        model.add(layers.Dense(100, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(80, input_dim=100, activation='relu'))
        model.add(layers.Dense(20, input_dim=80, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary())
        history = model.fit(xtrain, self.Y_train,
                            epochs=1000,
                            verbose=False,
                            validation_data=(xcv, self.Y_cv),
                            batch_size=10)

        clear_session()

        loss, accuracy = model.evaluate(xtrain, self.Y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(xcv, self.Y_cv, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    def guardarModelo(self, modelo, nombre):

        # Guardo el modelo.
        joblib.dump(modelo, './Interfaz/modelos/{}.pkl'.format(nombre))

    def cargarModelo(self, nombre):

        return joblib.load('./Interfaz/modelos/{}.pkl'.format(nombre))


class Index(QtWidgets.QMainWindow):
    def __init__(self):
        super(Index, self).__init__()
        # Cargamos el .ui file
        uic.loadUi('index.ui', self)
        # cargamos widgets
        self.setWidgets()
        # activamos botones
        self.activarBotones()
    # Aqui se declaran los widgets

    def setWidgets(self):
        self.about = self.findChild(QtWidgets.QPushButton, 'about')
        self.app = self.findChild(QtWidgets.QPushButton, 'app')
        self.train = self.findChild(QtWidgets.QPushButton, 'train')
        self.test = self.findChild(QtWidgets.QPushButton, 'test')
        self.downloads = self.findChild(QtWidgets.QPushButton, 'downloads')
        self.titulo = self.findChild(QLabel, 'titulo')
        self.descripcion = self.findChild(QLabel, 'descripcion')
        self.acceder = self.findChild(QtWidgets.QPushButton, 'acceder')
        self.icono = self.findChild(QtWidgets.QPushButton, 'icono')
        self.ljuancar = self.findChild(QLabel, 'ljuan')
        self.ladi = self.findChild(QLabel, 'ladi')
        self.lcarlos = self.findChild(QLabel, 'lcarlos')
        self.lrober = self.findChild(QLabel, 'lrober')
        self.juancar = self.findChild(QtWidgets.QPushButton, 'juan')
        self.adi = self.findChild(QtWidgets.QPushButton, 'adi')
        self.carlos = self.findChild(QtWidgets.QPushButton, 'carlos')
        self.rober = self.findChild(QtWidgets.QPushButton, 'rober')
        self.personalizar_boton(self.juancar, 'juancar.jpeg')
        self.personalizar_boton(self.adi, 'adi.jpeg')
        self.personalizar_boton(self.carlos, 'carlos.jpeg')
        self.personalizar_boton(self.rober, 'rober.jpg')
        self.acceder.setVisible(False)
        self.icono.setVisible(False)
        self.apagar_widgets(False)
    # aqui ponemos los eventos de los botones

    def activarBotones(self):
        self.about.clicked.connect(lambda: self.menuClicked('Startup Eateaser',
                                                            'Compañia encargada para sugerirte las mejores recetas Seremos tus aliados a la hora de cocinar.Nosotros te permitimos una aplicacion facilpara conocer la clasificacion de \n'
                                                            'tus platillos favoritos. Ademas tambien \clasificamos resetas y te enseñamos nuestros algoritmos',
                                                            False, True, ''))
        self.train.clicked.connect(lambda: self.menuClicked('Fase de Entrenamiento',
                                                            'Compañia encargada para sugerirte las mejores recetas.Seremos tus aliados a la hora de cocinar.\nNosotros te permitimos una aplicacion facil para conocer la clasificacion de \n'
                                                            'tus platillos favoritos. Ademas tambien \clasificamos resetas y te enseñamos nuestros algoritmos',
                                                            True, False, 'train.png'))
        self.test.clicked.connect(lambda: self.menuClicked('Fase de Testeo',
                                                           'Compañia encargada para sugerirte las mejores recetas.\Seremos tus aliados a la hora de cocinar.\nNosotros te permitimos una aplicacion facil\npara conocer la clasificacion de \n'
                                                           'tus platillos favoritos. Ademas tambien \clasificamos resetas y te enseñamos nuestros algoritmos',
                                                           True, False, 'test.png'))
        self.app.clicked.connect(lambda: self.menuClicked('Aplicación',
                                                          'Compañia encargada para sugerirte las mejores recetas.Seremos tus aliados a la hora de cocinar.Nosotros te permitimos una aplicacion facil para conocer la clasificacion de '
                                                          'tus platillos favoritos. Ademas tambien clasificamos resetas y te enseñamos nuestros algoritmos',
                                                          True, False, 'app.png'))
        self.downloads.clicked.connect(lambda: self.menuClicked('Descargar',
                                                                'Compañia encargada para sugerirte las mejores recetas.Seremos tus aliados a la hora de cocinar.Nosotros te permitimos una aplicacion facil para conocer la clasificacion de '
                                                                'tus platillos favoritos. Ademas tambien clasificamos resetas y te enseñamos nuestros algoritmos',
                                                                True, False, 'descarga.png'))
        self.acceder.clicked.connect(self.openWindow)

    def menuClicked(self, titulo, descripcion, acceder, widgets, dir):
        self.titulo.setText(titulo)
        self.descripcion.setText(descripcion)
        self.acceder.setVisible(acceder)
        self.apagar_widgets(widgets)
        self.state = titulo
        if(widgets == False):
            self.icono.setVisible(True)
            self.icono.setIcon(QIcon('imagenes/' + dir))
            self.icono.setIconSize(QSize(200, 200))
            self.icono.setStyleSheet('background-color:transparent;')
        else:
            self.icono.setVisible(False)

    def apagar_widgets(self, boolean):
        self.carlos.setVisible(boolean)
        self.juancar.setVisible(boolean)
        self.adi.setVisible(boolean)
        self.rober.setVisible(boolean)
        self.ladi.setVisible(boolean)
        self.lrober.setVisible(boolean)
        self.ljuancar.setVisible(boolean)
        self.lcarlos.setVisible(boolean)

    def personalizar_boton(self, boton, nombre):
        boton.setIcon(QIcon('imagenes/' + nombre))
        boton.setIconSize(QSize(200, 200))
        boton.setStyleSheet('background-color:transparent;')

    def openWindow(self):
        if self.state == 'Fase de Entrenamiento':
            self.gui = Train()
            self.gui.setWindowIcon(QtGui.QIcon('imagenes/chef-logo.ico'))

            self.gui.show()
            self.gui.showMaximized()
            self.close()
        if self.state == 'Fase de Testeo':
            self.gui = Test()
            self.gui.setWindowIcon(QtGui.QIcon('imagenes/chef-logo.ico'))

            self.gui.show()
            self.gui.showMaximized()
            self.close()
        if self.state == 'Aplicación':
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.gui = App()
            self.gui.setWindowIcon(QtGui.QIcon('imagenes/chef-logo.ico'))

            self.gui.show()
            self.gui.showMaximized()
            QApplication.restoreOverrideCursor()
            self.close()
        if self.state == 'Descargar':
            # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.gui = Download()
            self.gui.setWindowIcon(QtGui.QIcon('imagenes/chef-logo.ico'))
            self.gui.show()
            self.gui.showMaximized()
            # QApplication.restoreOverrideCursor()
            self.close()
    def volver_(self):
        self.gui = Index()
        self.gui.show()
        self.gui.showMaximized()
        self.close


class Train(Index):
    def __init__(self):
        super().__init__()
        uic.loadUi('train.ui', self)
        self.setWindowTitle("Eat Easer Train page")
        self.setWindowIcon(QIcon("imagenes/EatEaser-Logo.png"))
        self.cbcategoria =self.findChild(QComboBox,'comboBox')
        self.cbcategoria.addItems(os.listdir('recetastextos/'))
        self.varableSeleccionCarpetaGuardarModelo = ""
        self.setWidgets()
        self.df = pd.DataFrame()
        
        self.switchButtons()
    def setWidgets(self):
        
        self.seleccionados = []
        self.checkboxes = []
        
        self.seleccionlayout=self.findChild(QHBoxLayout,'box')
        self.anadir =self.findChild(QPushButton,'anadir')
        self.nuevo =self.findChild(QPushButton,'nuevo')
       
        
        
        self.volver =self.findChild(QPushButton,'volver')
        self.ltitulo  =self.findChild(QLabel,'ltitulo')
        self.ldescrip =self.findChild(QLabel,'ldescripcion')
        self.vista  =self.findChild(QLabel,'lvista')
        self.btn_svm=self.findChild(QPushButton,'svm')
        self.btn_rf =self.findChild(QPushButton,'rforest')
        self.btn_mr=self.findChild(QPushButton,'regression')
        self.btnalgoritmo=self.findChild(QPushButton,'play')
        self.formguardar =self.findChild(QLineEdit,'guardarform')
        self.borrar =self.findChild(QPushButton,'borrar')
        self.path_btn =self.findChild(QPushButton,'search_path')
        self.btn_guardar =self.findChild(QPushButton,'guardar_modelo')
    def switchButtons(self):
        # eventos de botones
        self.anadir.clicked.connect(self.aniadir_boton)
        self.borrar.clicked.connect(self.eliminar_boton)
        self.path_btn.clicked.connect(self.aniadir_directorio)
        self.btn_guardar.clicked.connect(self.guardarModelo)
        self.btn_svm.clicked.connect(
            lambda: (self.informacion('Algoritmo SVM', 'El entrenamiento de support vector machine se asemeja a resolver un problema de optimización cuadrática para ajustar un hiperplano que minimice el margen flexible entre las clases. El número de características transformadas está determinado por el número de vectores de soporte'), self.cambiar_algoritmo("SVM")))
        self.btn_mr.clicked.connect(
            lambda: (self.informacion('Algoritmo Multinomial Regression', 'la regresión logística multinomial generaliza el método de regresión logística para problemas multiclase, es decir, con más de dos posibles resultados discretos.1​ Es decir, se trata de un modelo que se utiliza para predecir las probabilidades de los diferentes resultados posibles de una distribución categórica como variable dependiente, dado un conjunto de variables independientes (que pueden ser de valor real, valor binario, categórico-valorado, etc.)'), self.cambiar_algoritmo("MR")))
        self.btn_rf.clicked.connect(
            lambda: (self.informacion('Algoritmo Random Forest', 'El algoritmo de bosque aleatorio es una extensión del método de embolsado, ya que utiliza tanto el embolsado como la aleatoriedad de características para crear un bosque de árboles de decisión no correlacionado. La aleatoriedad de características, también conocida como empaquetado de características o “ el método del subespacio aleatorio ”, genera un subconjunto aleatorio de características, lo que garantiza una baja correlación entre los árboles de decisión.'), self.cambiar_algoritmo("RF")))
        self.btnalgoritmo.clicked.connect(self.vista_previa)
        self.nuevo.clicked.connect(self.aniadir_categoria)
        self.volver.clicked.connect(self.volver_)
    def cambiar_algoritmo(self, nombre):
        self.algoritmo_clicked = nombre

    def informacion(self, titulo, descripcion):
        self.ltitulo.setText(titulo)
        self.ldescrip.setText(descripcion)
    # FUNCION PARA REALIZAR EL ALGORITMO

    def crearDF(self):

        df = pd.DataFrame()
        df['receta'] = None
        df['clasif'] = None
        procesarDocs = ProcesarDocumentos()
        listaTextosCarpeta = procesarDocs.lectura()
        diccionarioCarpetas = {'Carpeta Arroz': 0, 'Carpeta Bebidas': 1, 'Carpeta Carnes': 2,
                               'Carpeta Marisco': 3, 'Carpeta Pasta': 4, 'Carpeta Pescados': 5,
                               'Carpeta Platos Menores': 6, 'Carpeta Verduras': 7}

        print("++++++++++++++++++++++++++ {} ++++++++++++++++++++++++++++++".format(self.seleccionados))
        print(self.seleccionados[0])
        seleccion = []
        for i in range(len(self.seleccionados)):
            seleccion.append(diccionarioCarpetas[self.seleccionados[i]])

        for index, content in enumerate(listaTextosCarpeta):
            if index in seleccion:
                for i in range(len(content)):
                    text = procesarDocs.tratamientoTextos(
                        listaTextosCarpeta[index][i])
                    df = df.append(
                        {'receta': text, 'clasif': index}, ignore_index=True)

        print(df['clasif'].unique())

        return df

    def vista_previa(self):
        self.vista.setText('')
        i = 0
        self.total_archivos = 0

        carpetas = ''
        # verificamos si hay seleccionados

        if len(self.seleccionados) == 0 or self.ltitulo.text() == 'Nombre Algoritmo':

            self.mensaje_error('Campos vacios.')
        else:

            self.df1 = self.crearDF()
            modelo = modelosTFIDF(self.df1, 7000)

            if(self.algoritmo_clicked == "SVM"):
                # self.seleccionados
                self.modeloEntrenadoFinal = modelo.Entrenar_SVM()
                print('SVM')
            elif(self.algoritmo_clicked == "MR"):
                self.modeloEntrenadoFinal = modelo.Entrenar_RegresionMultinomial()
                print('MR')
            elif(self.algoritmo_clicked == "RF"):
                self.modeloEntrenadoFinal = modelo.Entrenar_RF()
                print('RF')
            self.vectorizer=modelo.vect
            print(self.df1.head())

            print("El modelo ha sido entrenado correctamente! :)")

            # verificamos si hay algoritmo seleccionado
            for i in self.seleccionados:

                size = len(os.listdir('recetastextos/' + i))

                self.total_archivos = size + self.total_archivos
                texto = i + ': ' + str(size) + ' archivos\n'

                carpetas = carpetas+'\n'+texto

        # le añado todos los que esten en listbox
        self.vista.setText(carpetas+'\n'+'TOTAL: ' + ': ' +
                           str(self.total_archivos) + ' archivos\n')

    def mensaje_error(self, mensaje):
        QMessageBox.critical(
            self,
            "Error",
            mensaje,
            buttons=QMessageBox.Discard | QMessageBox.NoToAll | QMessageBox.Ignore,
            defaultButton=QMessageBox.Discard,
        )

    def aniadir_boton(self):
        self.add = QCheckBox(self.cbcategoria.currentText())
        if any(i == self.cbcategoria.currentText() for i in self.seleccionados):
            self.mensaje_error(
                self.cbcategoria.currentText()+' ya esta seleccionada.')
        else:

            self.seleccionados.append(str(self.cbcategoria.currentText()))
            self.add.setStyleSheet('font-family:NSimsun')

            self.seleccionlayout.addWidget(self.add)
            self.checkboxes.append(self.add)
        print(self.seleccionados)

    def eliminar_boton(self):

        i = 0
        for c in self.checkboxes:
            if c.isChecked() == True:

                c.deleteLater()
                self.checkboxes.pop(i)
                self.seleccionados.pop(i)
                print(c.text())

            i = i+1

    def aniadir_categoria(self):
        r = QFileDialog.getExistingDirectory(
            self, "Select Directory", directory=os.getcwd())
        print(os.listdir(r))
        print(r)
        ultimo = r.split('/')[-1]
        print('ult', ultimo)
        # recorremos cada file del nuevo directorio
        for file_name in os.listdir(r):
            source = r + '/' + file_name
            destination = 'recetastextos/'+ultimo+'/' + file_name
            print('se va al destino', destination)
            # si existe el archivo de source lo movemos al destino

            if os.path.exists('recetastextos/'+ultimo) == False:
                os.makedirs('recetastextos/'+ultimo)
                shutil.move(source, destination)
                print('Moved:', file_name)
            else:
                # aqui va a haber un error
                shutil.move(source, destination)
                print('Moved:', file_name)
        # actualizamos el combobox
        self.cbcategoria.clear()
        self.cbcategoria.addItems(os.listdir('recetastextos/'))

    def aniadir_directorio(self):
        r = QFileDialog.getExistingDirectory(
            self, "Select Directory", directory=os.getcwd())
        self.varableSeleccionCarpetaGuardarModelo = r
       
    

    def guardarModelo(self, modeloEntrenado):
        if(self.varableSeleccionCarpetaGuardarModelo == ""):
            #print("no hay ruta")
            self.mensaje_error("No hay una ruta seleccionada")
        elif(self.formguardar.text() == ""):
            #print("no hay nombre de archivo")
            self.mensaje_error("Pon un nombre al archivo que se va a guardar")
        else:
            rutaGuardarModelo = self.varableSeleccionCarpetaGuardarModelo + \
                "/" + self.formguardar.text() + ".pkl"
            joblib.dump(self.modeloEntrenadoFinal, rutaGuardarModelo)
            print(rutaGuardarModelo)
            rutaGuardarVect = self.varableSeleccionCarpetaGuardarModelo + \
                "/" + self.formguardar.text() + "_vect.pkl"
            joblib.dump(self.vectorizer,rutaGuardarVect)
            
class Test_(Index):
    def __init__(self):
        super().__init__()
        uic.loadUi('test.ui', self)
       
        self.setWindowTitle("Eat Easer Test page")
        # variables globales
        self.nombrecarpeta = ''
        self.info = self.Informacion()
        self.varableRutaModeloEntrenado = ""
        self.nombrecarpetaTestosTest = ""
        self.cbcategoria = self.findChild(QLineEdit,'lineEdit')
        self.cbcategoria.setEnabled(False)
        self.btn_seleccion_modelo=self.findChild(QPushButton,'selectmodelo')
        self.nuevo=self.findChild(QPushButton,'aniadir')
        self.btnalgoritmo=self.findChild(QPushButton,'play')
        self.ltitulo=self.findChild(QLabel,'titulo')
        self.ltitulo=self.findChild(QLabel,'descripcion')
        # eventos de botones
        self.btn_seleccion_modelo.clicked.connect(
            lambda: self.recuperarRutaModeloEntrenado())
        self.nuevo.clicked.connect(self.aniadir_categoria)
        self.retorno=self.findChild(QPushButton,'volver')
        self.retorno.clicked.connect(self.volver_)
        self.btn_seleccion_modelo.clicked.connect(
            lambda: self.informacion('Modelo Seleccionado', 'Estos son sus archivos:'))
        self.btnalgoritmo.clicked.connect(self.vista_previa)
        self.grafico=self.findChild(QHBoxLayout,'horizontalLayout')
        self.tableWidget=QTableWidget()
        self.grafico.addWidget(self.tableWidget)
        self.vista =self.findChild(QLabel,'vista')
    def informacion(self, titulo, descripcion):
         self.ltitulo.setText(titulo)
         self.ldescrip.setText(descripcion)

    def recuperarRutaModeloEntrenado(self):
         r = QFileDialog.getOpenFileName(
             parent=None, caption='Select Directory', directory=os.getcwd(), filter='Pickle files (*.pkl)')
         direct=self.findChild(QLabel,'direccion')
         direct.setText(r)
         self.varableRutaModeloEntrenado = r[0]
         # print(self.varableRutaModeloEntrenado)

    def cargarModeloTest(self):
         if(self.varableRutaModeloEntrenado != ""):
             print("modelo cargado")

             self.modelo_entrenado = joblib.load(
                 self.varableRutaModeloEntrenado)
             self.tableWidget.setRowCount(
                 len(os.listdir(self.cbcategoria.placeholderText())))
             self.tableWidget.setColumnCount(3)
             self.info.ruta = os.listdir(self.cbcategoria.placeholderText())
             self.info.carpeta_seleccionada = self.cbcategoria.placeholderText()
             self.tableWidget.setHorizontalHeaderLabels(
                 ["Texto", "Categoria", "Ver Texto"])
             header = self.tableWidget.horizontalHeader()
             header.setSectionResizeMode(0, QHeaderView.Stretch)
             header.setSectionResizeMode(1, QHeaderView.Stretch)
             header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
             self.nombrecarpetaTestosTest = "".join(
                 self.cbcategoria.placeholderText())
             self.nombrecarpeta = self.cbcategoria.placeholderText().split(
                 '/')[-1]

    def setData(self):

         i = 0

         for key in os.listdir(self.cbcategoria.placeholderText()):
             boton = QPushButton()
             self.ver.addButton(boton)
             self.ver.setId(boton, i)
             boton.setIcon(QIcon('imagenes/ojo.png'))
             self.tableWidget.setItem(i, 0, QTableWidgetItem(key))
             self.tableWidget.setItem(i, 1, QTableWidgetItem(
                 "{}".format(self.prediccion[i])))
             self.tableWidget.setCellWidget(i, 2, boton)
             i = i+1

         self.tableWidget.resizeColumnsToContents()
         self.tableWidget.resizeRowsToContents()

         self.tableWidget.show()

    def aniadir_categoria(self):
         r = QFileDialog.getExistingDirectory(
             self, "Select Directory", directory=os.getcwd())
         self.cbcategoria.setPlaceholderText(r)



    def mensaje_error(self, mensaje):
         QMessageBox.critical(
             self,
             "Error",
             mensaje,
             buttons=QMessageBox.Discard | QMessageBox.NoToAll | QMessageBox.Ignore,
             defaultButton=QMessageBox.Discard,
         )

    def cargarDF_Completo(self):
         import pandas as pd
         df = pd.DataFrame()
         df['receta'] = None
         df['clasif'] = None
         procesarDocs = ProcesarDocumentos()
         listaTextosCarpeta = procesarDocs.lectura()
         for index, content in enumerate(listaTextosCarpeta):
             for i in range(len(content)):
                 text = procesarDocs.tratamientoTextos(
                     listaTextosCarpeta[index][i])
                 df = df.append(
                     {'receta': text, 'clasif': index}, ignore_index=True)
         return df

    def vista_previa(self):
         self.vista.setText('')
         i = 0
         self.total_archivos = 0

         carpetas = ''
         # verificamos si hay seleccionados

         if self.cbcategoria.placeholderText() == '' or self.ltitulo.text() == 'Nombre Algoritmo':

             self.mensaje_error('Campos vacios.')
         else:
             self.cargarModeloTest()
             modelo = self.modelo_entrenado
             numeroFeature = modelo.n_features_in_
             df_completo = self.cargarDF_Completo()
             rutaCarpetaTesting = self.nombrecarpetaTestosTest + \
                 "/"  # "c:/ddjashdashdjkash/../carpeta testing"
             mod = modelosTFIDF(df_completo, numeroFeature)
             print("\n \n \n \n \n Ruta: {} \n \n \n \n \n".format(
                 rutaCarpetaTesting))
             prediccion = mod.predecir_Carpeta(rutaCarpetaTesting, modelo)
             print("------------------------------- \n {} \n------------------------------------------".format(prediccion))
             diccionario = {0: "Arroz", 1: "Bebida", 2: "Carne", 3: "Marisco",
                            4: "Pasta", 5: "Pescado", 6: "Platos menores", 7: "Verdura"}
             resultado = []
             for i in prediccion:
                 resultado.append(diccionario[i])
             self.prediccion = resultado
             self.setData()
             # carpeta de entrenamiento
             # Vectorizer
             # usar metodo para predecir
             # si inicializamos el TFIDF hay q mandarle un df le mandamos uno con todo a piñon?
             #mod.predecir_Carpeta(self.nombrecarpetaTestosTest, varableRutaModeloEntrenado)

             size = len(os.listdir(self.cbcategoria.placeholderText()))

             self.total_archivos = size
             texto = self.nombrecarpeta + ': ' + str(size) + ' archivos\n'

             # le añado todos los que esten en listbox
             self.vista.setText(texto+'\n'+'TOTAL: ' + ': ' +
                                str(self.total_archivos) + ' archivos\n')
    class Informacion(Index):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("EatEaser-Visualizar Texto")
            scategorias = 'font-family:"NSimSun";font-size:20px;border:1px solid black;border-radius:12px;'
            stexto = 'line-height: 0.9;font-family:"NSimSun";font-size:24px;background-color:white;border:1px solid black;text-align:justify;text-transform:capitalize;padding:20px;'
            sventana = 'background-color:black;color:white;font-family:"NSimSun";font-size:20px;text-align:center;'
            self.layout = QGridLayout()
            # self.varableRutaModeloEntrenado=""
            self.nombre = QLabel('Nombre')
            self.categoria = QLabel('Categoria')
            self.texto = QLabel('Texto')
            self.informacion = QScrollArea()
            self.texto.setWordWrap(True)
            self.texto.setStyleSheet(stexto)
            self.categoria.setStyleSheet(scategorias)
            self.nombre.setStyleSheet(scategorias)
            self.id_ventana = QLabel('Visualizacion de texto')
            self.id_ventana.setStyleSheet(sventana)
            self.informacion.setWidget(self.texto)
            self.informacion.setWidgetResizable(True)
            self.layout.addWidget(self.id_ventana, 0, 0, 1, 4)
            self.layout.addWidget(self.nombre, 1, 0, 1,
                                  1, QtCore.Qt.AlignCenter)
            self.layout.addWidget(self.categoria, 1, 1, 1,
                                  1, QtCore.Qt.AlignCenter)
            self.layout.addWidget(self.informacion, 2, 0, 6, 4)

            self.setLayout(self.layout)
            self.ruta = []
            self.carpeta_seleccionada = ''
        
            

class Test(Index):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eat Easer Test page")
        self.setWindowIcon(QIcon("imagenes/EatEaser-Logo.png"))
        # variables globales
        self.nombrecarpeta = ''
        self.info = self.Informacion()
        self.varableRutaModeloEntrenado = ""
        self.nombrecarpetaTestosTest = ""

        # layout grande
        self.layout = QGridLayout()
        # partes del layout grande
        self.izqlayout = QGridLayout()
        self.derlayout = QGridLayout()
        # stylesheet
        scategoria = "font-family:'Bahnschrift Light';font-size:24px;letter-spacing:3px;padding:0%;padding:5px;"
        scbcategoria = "color :black;background-color:white;border-bottom:3px solid black;font-weight:lighter;font-size:22px;font-family:'Bahnschrift Light';letter-spacing:3px;"
        sbtnruta = 'QPushButton{background-color:transparent;border:1px solid transparent}QPushButton:hover{border:1px solid black;border-radius:12px;}'
        sbotones = 'QPushButton{border:transparent;background-color:transparent;}QPushButton:hover{border:2px solid black;border-radius:12px;}'
        salgoritmo = "font-family:'Bahnschrift Light';font-size:24px;letter-spacing:3px;padding:0%"
        sbtnalgoritmo = 'QPushButton{color:white;border-radius:12px;background-color:black;margin:0;font-family:"Bahnschrift Light";font-size:24px;}QPushButton:hover{color:black;background-color:transparent;border:2px solid black;}'
        sform = "font-family:'Bahnschrift Light';font-style:italic;font-size:24px;letter-spacing:3px;padding:0%"
        sinfo = 'background-color:white;border-radius:12px;border:1px white;'
        stextos_derecha = 'font-family:"NSimSun";font-size:24px;overflow:hidden;white-space: nowrap;'
        sretorno = 'font-family:"NSimSun";font-size:24px;overflow:hidden;white-space: nowrap;color:white;background-color:black;'
        # grid de la ruta
        self.rutalayout = QGridLayout()
        # labels de ruta
        self.lcategoria = QLabel('Textos a clasificar')
        self.lcategoria.setStyleSheet(scategoria)
        # combobox de ruta
        self.cbcategoria = QLineEdit()
        self.cbcategoria.setEnabled(False)
        self.cbcategoria.setFixedSize(800, 40)
        self.cbcategoria.setStyleSheet(scbcategoria)
        # botones de ruta

        self.nuevo = QPushButton()
        self.nuevo.setIcon(QIcon('imagenes/add.png'))
        self.nuevo.setFixedSize(QtCore.QSize(40, 40))
        # estilizamos los botones
        self.nuevo.setStyleSheet(sbtnruta)

        # aniadimos al layout de ruta
        self.rutalayout.addWidget(self.lcategoria, 0, 0, 1, 1)
        self.rutalayout.addWidget(self.cbcategoria, 0, 1, 1, 1)
        self.rutalayout.addWidget(self.nuevo, 0, 2, 1, 1)

        # grid de algoritmos
        self.algoritmolayout = QGridLayout()
        # labels algoritmo
        self.lalgoritmo = QLabel('Algoritmo:')
        self.lalgoritmo.setStyleSheet(salgoritmo)
        # botones de algorimos
        self.btn_seleccion_modelo = QPushButton('Seleccionar Modelo')
        self.btnalgoritmo = QPushButton()
        # estilizamos botones
        self.btn_seleccion_modelo.setFixedSize(QtCore.QSize(400, 80))
        self.btn_seleccion_modelo.setStyleSheet(sbtnalgoritmo)
        size = QSize(50, 50)
        self.btnalgoritmo.setIconSize(size)
        self.btnalgoritmo.setStyleSheet(sbotones)
        self.btnalgoritmo.setIcon(QIcon('imagenes/boton-de-play.png'))
        self.btnalgoritmo.setFixedSize(QtCore.QSize(80, 80))

        # aniadimos al grid de algoritmos
        self.algoritmolayout.addWidget(self.lalgoritmo, 0, 0, 1, 4)
        self.algoritmolayout.addWidget(self.btn_seleccion_modelo, 1, 1, 2, 1)
        self.algoritmolayout.addWidget(self.btnalgoritmo, 1, 3, 2, 1)

        # grjd guardar
        self.guardar = QGridLayout()

        # botones de grid guardar
        self.path_btn = QPushButton('')
        self.btn_guardar = QPushButton()

        # estilizamos botones
        self.path_btn.setIcon(QIcon('imagenes/lupa.png'))
        self.btn_guardar.setIcon(QIcon('imagenes/guardar-el-archivo.png'))
        self.btn_guardar.setStyleSheet(sbotones)
        self.path_btn.setStyleSheet(sbotones)

        # eventos de botones
        self.btn_seleccion_modelo.clicked.connect(
            lambda: self.recuperarRutaModeloEntrenado())
        # self.path_btn.clicked.connect()
        # self.btn_guardar.clicked.connect(self.recuperarModeloEntrenado)
        self.nuevo.clicked.connect(self.aniadir_categoria)
        # form del grid guardar
        self.formguardar = QLineEdit()
        self.lform = QLabel("Guardar Resultados:")
        self.lform.setStyleSheet(sform)

        # aniadimos los widgets a guardar
        self.guardar.addWidget(self.lform, 0, 0, 1, 1)
        self.guardar.addWidget(self.formguardar, 0, 1, 1, 1)
        self.guardar.addWidget(self.btn_guardar, 0, 2)
        self.guardar.addWidget(self.path_btn, 0, 3)

        # grid del grafico

        self.grafico = QVBoxLayout()

        # agregamos la tabla
        self.tableWidget = QTableWidget()
        self.btn_seleccion_modelo.clicked.connect(
            lambda: self.informacion('Modelo Seleccionado', 'Estos son sus archivos:'))
        self.btnalgoritmo.clicked.connect(self.vista_previa)

        self.grafico.addWidget(self.tableWidget)

        # boton de retorno izquierdo
        self.retorno = QPushButton(u"\u2190" + ' Main Page/ Test')
        self.retorno.setStyleSheet(sretorno)
        self.retorno.clicked.connect(self.volver_)
        # aniadimos los layouts al lado izq
        self.izqlayout.addWidget(self.retorno, 0, 0)
        self.izqlayout.addLayout(self.rutalayout, 1, 0)
        self.izqlayout.addLayout(self.algoritmolayout, 2, 0)
        self.izqlayout.addLayout(self.grafico, 3, 0)
        self.izqlayout.addLayout(self.guardar, 4, 0)
        # estilizamos los layouts
        self.izqlayout.setRowStretch(0, 1)
        self.izqlayout.setRowStretch(1, 1)
        self.izqlayout.setRowStretch(2, 1)
        self.izqlayout.setRowStretch(3, 2)
        self.izqlayout.setRowStretch(4, 1)

        # zona derecha del layout labels
        self.linfo = QPushButton()
        self.ltitulo = QLabel('Nombre Algoritmo')
        self.ldescrip = QLabel('Descripcion Algoritmo')
        self.vista = QLabel('Vista Algoritmo')
        self.fondo = QLabel()
        # estilizar labels
        self.linfo.setIcon(QIcon('imagenes/informacion.png'))
        self.linfo.setStyleSheet(sinfo)
        self.linfo.setFixedSize(QtCore.QSize(400, 80))
        size = QSize(50, 50)
        self.linfo.setIconSize(size)
        self.fondo.setStyleSheet(sinfo)

        # aniadimos widgets en lado derecho
        self.derlayout.addWidget(self.fondo, 0, 0, 6, 1)
        self.derlayout.addWidget(
            self.linfo, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(
            self.ltitulo, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(
            self.ldescrip, 2, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(
            self.vista, 3, 0, 3, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.rowStretch(1)

        # aniadimos los layouts al total
        self.layout.addLayout(self.izqlayout, 0, 0)
        self.layout.addLayout(self.derlayout, 0, 1)
        self.layout.setColumnStretch(0, 3)
        self.layout.setColumnStretch(1, 1)

        # estilizamos zona derecha
        self.ltitulo.setStyleSheet(stextos_derecha)
        self.ldescrip.setStyleSheet(stextos_derecha)
        self.vista.setStyleSheet(stextos_derecha)

        self.setLayout(self.layout)
        self.ver = QButtonGroup()
        self.ver.buttonClicked[int].connect(self.info.ver_)

    class Informacion(Index):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("EatEaser-Visualizar Texto")
            scategorias = 'font-family:"NSimSun";font-size:20px;border:1px solid black;border-radius:12px;'
            stexto = 'line-height: 0.9;font-family:"NSimSun";font-size:24px;background-color:white;border:1px solid black;text-align:justify;text-transform:capitalize;padding:20px;'
            sventana = 'background-color:black;color:white;font-family:"NSimSun";font-size:20px;text-align:center;'
            self.layout = QGridLayout()
            # self.varableRutaModeloEntrenado=""
            self.nombre = QLabel('Nombre')
            self.categoria = QLabel('Categoria')
            self.texto = QLabel('Texto')
            self.informacion = QScrollArea()
            self.texto.setWordWrap(True)
            self.texto.setStyleSheet(stexto)
            self.categoria.setStyleSheet(scategorias)
            self.nombre.setStyleSheet(scategorias)
            self.id_ventana = QLabel('Visualizacion de texto')
            self.id_ventana.setStyleSheet(sventana)
            self.informacion.setWidget(self.texto)
            self.informacion.setWidgetResizable(True)
            self.layout.addWidget(self.id_ventana, 0, 0, 1, 4)
            self.layout.addWidget(self.nombre, 1, 0, 1,
                                  1, QtCore.Qt.AlignCenter)
            self.layout.addWidget(self.categoria, 1, 1, 1,
                                  1, QtCore.Qt.AlignCenter)
            self.layout.addWidget(self.informacion, 2, 0, 6, 4)

            self.setLayout(self.layout)
            self.ruta = []
            self.carpeta_seleccionada = ''

        def ver_(self, list):
            with open(self.carpeta_seleccionada+'/'+self.ruta[list], "r") as archivo:
                for linea in archivo:
                    resultado = linea

            self.texto.setText(resultado)
            self.nombre.setText(self.ruta[list])
            self.gui = self
            self.gui.show()
            width = 900
            height = 500
            # setting  the fixed size of window
            self.gui.setFixedSize(width, height)

    def informacion(self, titulo, descripcion):
        self.ltitulo.setText(titulo)
        self.ldescrip.setText(descripcion)

    def recuperarRutaModeloEntrenado(self):
        r = QFileDialog.getOpenFileName(
            parent=None, caption='Select Directory', directory=os.getcwd(), filter='Pickle files (*.pkl)')
        self.varableRutaModeloEntrenado = r[0]
        # print(self.varableRutaModeloEntrenado)

    def cargarModeloTest(self):
        if(self.varableRutaModeloEntrenado != ""):
            print("modelo cargado")

            self.modelo_entrenado = joblib.load(
                self.varableRutaModeloEntrenado)
            self.tableWidget.setRowCount(
                len(os.listdir(self.cbcategoria.placeholderText())))
            self.tableWidget.setColumnCount(3)
            self.info.ruta = os.listdir(self.cbcategoria.placeholderText())
            self.info.carpeta_seleccionada = self.cbcategoria.placeholderText()
            self.tableWidget.setHorizontalHeaderLabels(
                ["Texto", "Categoria", "Ver Texto"])
            header = self.tableWidget.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.nombrecarpetaTestosTest = "".join(
                self.cbcategoria.placeholderText())
            self.nombrecarpeta = self.cbcategoria.placeholderText().split(
                '/')[-1]

    def setData(self):

        i = 0

        for key in os.listdir(self.cbcategoria.placeholderText()):
            boton = QPushButton()
            self.ver.addButton(boton)
            self.ver.setId(boton, i)
            boton.setIcon(QIcon('imagenes/ojo.png'))
            self.tableWidget.setItem(i, 0, QTableWidgetItem(key))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(
                "{}".format(self.prediccion[i])))
            self.tableWidget.setCellWidget(i, 2, boton)
            i = i+1

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()

        self.tableWidget.show()

    def aniadir_categoria(self):
        r = QFileDialog.getExistingDirectory(
            self, "Select Directory", directory=os.getcwd())
        self.cbcategoria.setPlaceholderText(r)



    def mensaje_error(self, mensaje):
        QMessageBox.critical(
            self,
            "Error",
            mensaje,
            buttons=QMessageBox.Discard | QMessageBox.NoToAll | QMessageBox.Ignore,
            defaultButton=QMessageBox.Discard,
        )

    def cargarDF_Completo(self):
        import pandas as pd
        df = pd.DataFrame()
        df['receta'] = None
        df['clasif'] = None
        procesarDocs = ProcesarDocumentos()
        listaTextosCarpeta = procesarDocs.lectura()
        for index, content in enumerate(listaTextosCarpeta):
            for i in range(len(content)):
                text = procesarDocs.tratamientoTextos(
                    listaTextosCarpeta[index][i])
                df = df.append(
                    {'receta': text, 'clasif': index}, ignore_index=True)
        return df

    def vista_previa(self):
        self.vista.setText('')
        i = 0
        self.total_archivos = 0

        carpetas = ''
        # verificamos si hay seleccionados

        if self.cbcategoria.placeholderText() == '' or self.ltitulo.text() == 'Nombre Algoritmo':

            self.mensaje_error('Campos vacios.')
        else:
            self.cargarModeloTest()
            modelo = self.modelo_entrenado
            numeroFeature = modelo.n_features_in_
            df_completo = self.cargarDF_Completo()
            rutaCarpetaTesting = self.nombrecarpetaTestosTest + \
                "/"  # "c:/ddjashdashdjkash/../carpeta testing"
            mod = modelosTFIDF(df_completo, numeroFeature)
            #vectorizer=mod.vect
            
            print("\n \n \n \n \n Ruta: {} \n \n \n \n \n".format(
                rutaCarpetaTesting))
            prediccion = mod.predecir_Carpeta(rutaCarpetaTesting, modelo)
            print("------------------------------- \n {} \n------------------------------------------".format(prediccion))
            diccionario = {0: "Arroz", 1: "Bebida", 2: "Carne", 3: "Marisco",
                           4: "Pasta", 5: "Pescado", 6: "Platos menores", 7: "Verdura"}
            resultado = []
            for i in prediccion:
                resultado.append(diccionario[i])
            self.prediccion = resultado
            self.setData()
            # carpeta de entrenamiento
            # Vectorizer
            # usar metodo para predecir
            # si inicializamos el TFIDF hay q mandarle un df le mandamos uno con todo a piñon?
            #mod.predecir_Carpeta(self.nombrecarpetaTestosTest, varableRutaModeloEntrenado)

            size = len(os.listdir(self.cbcategoria.placeholderText()))

            self.total_archivos = size
            texto = self.nombrecarpeta + ': ' + str(size) + ' archivos\n'

            # le añado todos los que esten en listbox
            self.vista.setText(texto+'\n'+'TOTAL: ' + ': ' +
                               str(self.total_archivos) + ' archivos\n')


class WebScraping:
    def __init__(self, kw):
        self.keyword = kw
        self.listaNombres = []
        self.listaTiempos = []
        self.listaImagenes = []
        self.listaPrecios = []
        self.listaURL = []

    def conexionPaginaWebLidl(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(
            ChromeDriverManager().install(), chrome_options=options)
        urlLidl = "https://recetas.lidl.es/"
        driver.get(urlLidl)
        time.sleep(0.5)
        self.quitarCookiesLidl(driver)
        time.sleep(1)
        self.sacarInfoLidl(driver)
        return driver

    def conexionPaginaWebAhorraMas(self):

        urlAhorraMas = "https://www.ahorramas.com/buscador?q=" + self.keyword
        response = requests.get(urlAhorraMas)
        soup = BeautifulSoup(response.content, "html.parser")
        time.sleep(0.5)
        self.sacarInfoAhorraMas(soup)

    def quitarCookiesLidl(self, driver):
        # si sale el boton de las koockies lo acepta
        try:
            driver.find_element(
                By.CLASS_NAME, "cookie-alert-extended-button").click()
            print("Boton aceptar coquies seleccionado")
        except:
            print("No sale boton de aceptar coquies")

    def sacarInfoLidl(self, driver):
        # Para entrar en el alimento que nosotros queremos
        escribirIngrediente = driver.find_element(
            By.CLASS_NAME, "inputField.js_mIngredientSearchGroup-input")
        escribirIngrediente.send_keys(self.keyword)
        # hay que darle un poco de tiempo para que despues de escribir seleccione el enter sino no lo ejecuta bien
        time.sleep(2)
        escribirIngrediente.send_keys(Keys.RETURN)

        time.sleep(2)
        # PARA COGER EL NOMBRE DE LA RECETA DEL PRODUCTO
        todasRecetas = driver.find_element(
            By.CLASS_NAME, "oRecipeFeed-resultContainer.js_oRecipeFeed-resultContainer")
        time.sleep(1)
        nombre = todasRecetas.find_elements(
            By.CLASS_NAME, "mRecipeTeaser-title")
        self.listaNombres = []
        for n in nombre:
            self.listaNombres.append(n.text)
           

        # PARA COGER EL TIEMPO DEL PRUDUCTO EN COCINARSE
        tiempoReceta = todasRecetas.find_elements(By.CLASS_NAME, "mTimer-time")
        self.listaTiempos = []
        for t in tiempoReceta:
            self.listaTiempos.append(t.text)
        time.sleep(0.5)
        # PARA COGER LA IMAGEN DEL PRODUCTO
        src = todasRecetas.find_elements(
            By.CLASS_NAME, "picture-img.mRecipeTeaser-image.lazyloaded")
        self.listaImagenes = []
        for s in src:
            self.listaImagenes.append(s.get_attribute("src"))

        # PARA COGER LA URL DE LA RECETA
        href = todasRecetas.find_elements(By.CLASS_NAME, "mRecipeTeaser-link")
        self.listaURL = []
        for h in href:
            self.listaURL.append(h.get_attribute("href"))

    def sacarInfoAhorraMas(self, soup):
        # PARA COGER EL NOMBRE DEL PRODUCTO
        bodyInformacion = soup.find_all(class_="tile-body")
        self.listaNombres = []
        for np in bodyInformacion:
            nombreProducto = np.find(class_="link product-name-gtm")
            self.listaNombres.append(nombreProducto.text)
            print('Hola',nombreProducto.text)

        # PARA COGER EL PRECIO DEL PRODUCTO
        bodyInformacion = soup.find_all(class_="tile-body")
        self.listaPrecios = []
        for p in bodyInformacion:
            valorProducto = p.find(class_="value")
            # utilizamos strip para eliminar los espacios en blanco tanto delante como detras del precio
            self.listaPrecios.append(valorProducto.text.strip())
            # PARA COGER LA IMAGEN DEL PRODUCTO
            src = soup.find_all(class_="tile-image")
            self.listaImagenes = []
            for s in src:
                self.listaImagenes.append(s["src"])
                print(s["src"])

            # PARA COGER LA URL DEL PRODUCTO
            href = soup.find_all(class_="product-pdp-link")
            self.listaURL = []
            for h in href:
                # concatenamos la direccion de la pagina la cual no incluye el href al usar beautifulsoup
                self.listaURL.append("https://www.ahorramas.com" + h["href"])
            # Eliminamos los elementos duplicados ya que coge cada url duplicada
            # print(len(listaURL))
            listaURLSinDuplicados = []
            for url in self.listaURL:
                if url not in listaURLSinDuplicados:
                    listaURLSinDuplicados.append(url)
            # print(len(listaURLSinDuplicados))


class Download(Index):
    def __init__(self):
        super(Download, self).__init__()
        # Cargamos el .ui file
        uic.loadUi('download.ui', self)
        self.info = self.Informacion()
        self.grid = self.findChild(QGridLayout, 'grid_descargas')

        self.btn_descarga = self.findChild(QPushButton, 'descarga')
        self.enlace = self.findChild(QLineEdit, 'enlace')
        self.volver = self.findChild(QPushButton, 'volver')
        self.btn_descarga.clicked.connect(self.descargar)
        self.volver.clicked.connect(self.volver_)
        self.btn_group = QButtonGroup()
        self.btn_group.buttonClicked[int].connect(self.info.setTexto)

    class Informacion(QMainWindow):
        def __init__(self):
            super().__init__()
            # Cargamos el .ui file
            uic.loadUi('vista.ui', self)

            rc = RecursosAdicionales()
            self.texto = self.findChild(QLabel, 'texto')
            self.nombre = self.findChild(QLabel, 'nombre')

            self.copiar = self.findChild(QPushButton, 'copiar')
            self.copiar.clicked.connect(self.copiar_)

        def setTexto(self, id):
            with open('recetastextos/receta' + str(id) + '.txt', "r") as archivo:
                for linea in archivo:
                    resultado = linea
            self.texto.setText(resultado)
            self.nombre.setText('receta' + str(id) + '.txt')
            self.show()
            width = 900
            height = 500
            self.setFixedSize(width, height)

        def copiar_(self):
            clipboard.copy(self.texto.text())

   

    def descargar(self):
        col = 0
        fila = 0
        dp = ControladorVideo(self.enlace.text())
        dp.video()

        for i in range(4):
            # si es una lista vamos a rellenarlo todo

            if (col < 4):
                if dp.esLista() == True:
                    frame = QFrame()
                    self.grid.addWidget(frame, fila, col)

                    frame_grid = QVBoxLayout()

                    frame.setLayout(frame_grid)
                    grid_titulo = QVBoxLayout()
                    grid_boton = QVBoxLayout()
                    frame_grid.addLayout(grid_titulo)
                    titulo = QLabel(dp.cv.nombrevideo)
                    titulo.setStyleSheet(
                        'font-family:"Bahnschrift Light";font-size:16px;')
                    grid_titulo.addWidget(titulo)
                    frame_grid.addLayout(grid_boton)
                    ver = QPushButton('Ver texto')
                    self.btn_group.addButton(ver, dp._idvideo)
                    ver.setStyleSheet(
                        'background-color:black;color:white;font-family:"Californian FB";font-size:16px;border-radius:20px;')
                    ver.setFixedSize(100, 60)
                    grid_boton.addWidget(ver)
                    frame.setStyleSheet(
                        'background-color:white;border-radius:20px;')
                    col = col + 1
                else:
                    frame = QFrame()
                    self.grid.addWidget(frame, fila, col)

                    frame_grid = QVBoxLayout()

                    frame.setLayout(frame_grid)
                    grid_titulo = QVBoxLayout()
                    grid_boton = QVBoxLayout()
                    frame_grid.addLayout(grid_titulo)
                    titulo = QLabel(dp.nombrevideo)
                    titulo.setStyleSheet(
                        'font-family:"Bahnschrift Light";font-size:16px;')
                    grid_titulo.addWidget(titulo)
                    frame_grid.addLayout(grid_boton)
                    ver = QPushButton('Ver texto')
                    self.btn_group.addButton(ver, dp._idvideo)
                    ver.setStyleSheet(
                        'background-color:black;color:white;font-family:"Californian FB";font-size:16px;border-radius:20px;')
                    ver.setFixedSize(100, 60)
                    grid_boton.addWidget(ver)
                    frame.setStyleSheet(
                        'background-color:white;border-radius:20px;')

                    if col != 0:
                        titulo.hide()
                        ver.hide()
                    col = col + 1
            else:
                col = 0
                fila = fila + 1


class App(Index):
    def __init__(self):
        super(App, self).__init__()
        # Cargamos el .ui file
        uic.loadUi('app.ui', self)

        # ponemos unos estilos
        scategorias = 'border-radius:100px;}QPushButton:hover{border:4px solid black;}'

        # llamamos a los botones
        self.btnplatos = self.findChild(QtWidgets.QPushButton, 'btnplatos')
        self.btnplatos.setStyleSheet(
            "QPushButton{border-image:url(imagenes/platos.jpg);"+scategorias)
        self.btnplatos.setFixedSize(200, 200)
        self.btnverdura = self.findChild(QtWidgets.QPushButton, 'btnverdura')
        self.btnverdura.setStyleSheet(
            "QPushButton{border-image:url(imagenes/verdura.jpg);"+scategorias)
        self.btnverdura.setFixedSize(200, 200)
        self.btnarroz = self.findChild(QtWidgets.QPushButton, 'btnarroz')
        self.btnarroz.setStyleSheet(
            "QPushButton{border-image:url(imagenes/arroz.jpg);"+scategorias)
        self.btnarroz.setFixedSize(200, 200)
        self.btnpasta = self.findChild(QtWidgets.QPushButton, 'btnpasta')
        self.btnpasta.setStyleSheet(
            "QPushButton{border-image:url(imagenes/pasta.jpg);"+scategorias)
        self.btnpasta.setFixedSize(200, 200)
        self.btnmarisco = self.findChild(QtWidgets.QPushButton, 'btnmarisco')
        self.btnmarisco.setStyleSheet(
            "QPushButton{border-image:url(imagenes/marisco.jpg);"+scategorias)
        self.btnmarisco.setFixedSize(200, 200)
        self.btnpescado = self.findChild(QtWidgets.QPushButton, 'btnpescado')
        self.btnpescado.setStyleSheet(
            "QPushButton{border-image:url(imagenes/pescado.jpg);"+scategorias)
        self.btnpescado.setFixedSize(200, 200)
        self.btnbebidas = self.findChild(QPushButton, 'btnbebidas')
        self.btnbebidas.setStyleSheet(
            "QPushButton{border-image:url(imagenes/bebida.jpg);"+scategorias)
        self.btnbebidas.setFixedSize(200, 200)
        self.btncarne = self.findChild(QPushButton, 'btncarne')
        self.btncarne.setStyleSheet(
            "QPushButton{background-image:url(imagenes/carne.jpg);"+scategorias)
        self.btncarne.setFixedSize(200, 200)
        self.busqueda = self.findChild(QLineEdit, 'busqueda')
        self.btnbuscar = self.findChild(QPushButton, 'buscar')
        self.btnbuscar.setIcon(QIcon('imagenes/lupa.png'))
        self.busqueda.setStyleSheet(
            "QPushButton{border-radius:10px;border:1px solid black;background-color:transparent;")
        #self.volver = self.findChild(QPushButton, 'back')
        # self.volver.setIcon(QIcon('imagenes/menu.png'))
        self.txt_frame = self.findChild(QGridLayout, 'gridLayout_8')
        self.volver = self.findChild(QPushButton, 'volver')
        self.grupo_botones = QButtonGroup()
        self.grid_productos = self.findChild(QGridLayout, 'grid_productos')
       
        self.productos_lt=self.findChild(QHBoxLayout,'lt_productos')
        # agrego primer frame
        self.productos=QButtonGroup()
        self.ws = WebScraping('cebolla')
        self.buscar_recetas('cebolla')
        self.buscar_texto('Carpeta Arroz')
        # ponemos acciones a los botones
        self.btnbuscar.clicked.connect(
            lambda: self.buscar_recetas(self.busqueda.text()))
        self.btncarne.clicked.connect(lambda: self.buscar_recetas('carne'))
        self.btnpasta.clicked.connect(lambda: self.buscar_recetas('pasta'))
        self.btnpescado.clicked.connect(lambda: self.buscar_recetas('pescado'))
        self.btnmarisco.clicked.connect(lambda: self.buscar_recetas('marisco'))
        self.btnbebidas.clicked.connect(lambda: self.buscar_recetas('bebida'))
        self.btnplatos.clicked.connect(lambda: self.buscar_recetas('pan'))
        self.btnverdura.clicked.connect(lambda: self.buscar_recetas('lechuga'))
        self.productos.buttonClicked[int].connect(self.mostrar_pagina)
        self.volver.clicked.connect(self.volver_)
        # ponemos un default de recetas
       

        self.setStyleSheet('background-color:white;')
        self.show()
    def mostrar_pagina(self,id_):
        print('boton press')
        print(self.ws.listaURL[id_])
        webbrowser.open(self.ws.listaURL[id_]) 
        pass
    def buscar_texto(self, categoria):
        directorio = os.listdir('recetastextos/'+categoria)

        j = 0
        fila = 0
        for i, texto in enumerate(directorio):
            # quiero que sean 10 columnas

            if j < 10:

                self.boton = QPushButton(texto)
                self.boton.setStyleSheet(
                    'QPushButton{font-family:Lucida Bright;border-radius:5px;border:1px solid black;}QPushButton:hover{border:1px solid white;background-color:black;color:white;}')
                self.txt_frame.addWidget(self.boton, fila, j)
                j = j+1
            else:
                j = 0
                fila = fila+1

    def buscar_productos(self, producto):
        
        cont = 0
      
        self.ws.conexionPaginaWebAhorraMas()
        
        for i, element in enumerate(self.ws.listaNombres):
            if(cont==8):
                break
            print(element)
            # si la columna ya va a mas de uno
            
                
                
            lt=QVBoxLayout()
            self.productos_lt.addLayout(lt)
          
            imagen=QPushButton()
            
            nombre=QLabel()
            nombre.setWordWrap(True)
            descripcion=QLabel()
            lt.addWidget(imagen)
            lt.addWidget(nombre)
            lt.addWidget(descripcion)
        
            nombre.setText(element)
            nombre.setStyleSheet('font-family:Simsun;color:white;')
            descripcion.setText(self.ws.listaPrecios[i])
            descripcion.setStyleSheet('font-family:Lucida Bright;color:white;font-weight:bold;')
            print('seteando text')
            response = requests.get(self.ws.listaImagenes[i])
            if response.status_code == 200:
                with open("imagenes/sample_producto" + str(i) + ".jpg", 'wb') as f:
                    f.write(response.content)
            imagen.setStyleSheet(
                "border-image:url(imagenes/sample_producto" + str(i) + ".jpg);")
            imagen.setFixedSize(200, 200)
            self.productos.addButton(imagen,i)
                

            cont = cont + 1

    def buscar_recetas(self, categoria):
        col = 0
        fila = 2
        ws2 = WebScraping(categoria)
        ws2.conexionPaginaWebLidl()
        for i, element in enumerate(ws2.listaNombres):
            if col < 4:
                img = True
                try:
                    imagen = QPushButton('')
                    response = requests.get(ws2.listaImagenes[i])
                    if response.status_code == 200:
                        with open("imagenes/sample" + str(i) + ".jpg", 'wb') as f:
                            f.write(response.content)
                    imagen.setStyleSheet(
                        "border-image:url(imagenes/sample" + str(i) + ".jpg);border-radius:100%;")
                    imagen.setFixedSize(200, 200)

                except:
                    print('imagen no obtenida')
                    img = False
               
                if (img == True):
                    n = QFrame()
                    n2 = QFrame()
                    n3 = QFrame()
                    n4 = QFrame()
                    vlt = QVBoxLayout()
                    n.setLayout(vlt)
                    n.setMinimumHeight(470)
                    n.setMaximumHeight(470)
                    
                    
                    self.grid_productos.addWidget(n, fila, col)
                    # vlt2 = QVBoxLayout()
                    # n.layout().addWidget(vlt2)
                    vlt.addWidget(n2)
                    n2.setLayout(QVBoxLayout())
                    n3.setLayout(QVBoxLayout())
                    n4.setLayout(QHBoxLayout())
                    
                    # parte de arriba
                    n2.layout().addWidget(QLabel(element))
                   
                    n2.layout().addWidget(imagen)
                    n2.layout().setAlignment(QtCore.Qt.AlignHCenter)
                    n3.layout().addWidget(QLabel('Descripción'))
                    #n3.setMinimumHeight(300)
                    n3.layout().addWidget(QLabel(ws2.listaTiempos[i]))
                    n4.layout().addWidget(QLabel('♡'))
                   # n4.setMinimumHeight(300)
                    btn = QPushButton('')

                    btn.setIcon(QIcon('imagenes/exterior.png'))

                    btn.setIconSize(QSize(20, 20))
                    n4.layout().addWidget(btn)

                    # parte del medio
                    vlt.addWidget(n3)
                    vlt.addWidget(n4)
                    # parte de abajo
                    
                    n.setStyleSheet(
                        'border:1px solid black;font-family:"Segoe UI Semibold";font-size:16px;border-radius:20px;')
                    n2.setStyleSheet(
                        'background-color:white;text-decoration: underline;border:1px solid white;')
                    n3.setStyleSheet(
                        'background-color:white;text-decoration: underline;border:1px solid white;')
                    n4.setStyleSheet(
                        'background-color:white;text-decoration: underline;border:1px solid white;')
                    col = col + 1
            else:
                col = 0
                fila = fila + 1

        self.buscar_productos(categoria)


if __name__ == '__main__':

    app = QApplication(sys.argv)

    gui = Index()
    gui.setWindowIcon(QtGui.QIcon('imagenes/chef-logo.ico'))
    gui.show()
    gui.showMaximized()

    sys.exit(app.exec_())
    
