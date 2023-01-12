import os
import sys
import shutil
import urllib.request
import joblib
#imports de la interfaz
from PyQt5 import QtCore, QtWidgets, QtGui, uic
from PyQt5.QtCore import QSize
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


import glob
import os
import re
import requests
import pathlib
import sys
import logging
import json
import joblib
import warnings
import math
import random
import multiprocessing
from random import shuffle
import subprocess
import time


def install(package):
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", package])



try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
except ModuleNotFoundError:
    install("selenium")
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains

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
    from pydub.silence import split_on_silence
except:
    install("pydub")
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
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
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
except:
    install("nltk")
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
try:
    import pyrebase
except:
    install("pyrebase4")
    import pyrebase
try:   
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.rslp import RSLPStemmer
    nltk.download('rslp')
except:
    install("nltk")
    import nltk
    #nltk.download('punkt')
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.rslp import RSLPStemmer
    nltk.download('rslp')
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import ParameterGrid
    from sklearn.model_selection import GridSearchCV
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_blobs
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import KFold
except ModuleNotFoundError:
    install("scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import ParameterGrid
    from sklearn.model_selection import GridSearchCV
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_blobs
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import KFold
try:
    import numpy as np
    from scipy import stats
except ModuleNotFoundError:
    install("numpy")
    import numpy as np
    from scipy import stats
try:
    import tensorflow as tf
except ModuleNotFoundError:
    install("tensorflow")
    import tensorflow as tf
try:
    from keras.models import Sequential
    from keras import layers
except ModuleNotFoundError:
    install("keras")
    from keras.models import Sequential
    from keras import layers
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    install("matplotlib")
    import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ModuleNotFoundError:
    install("seaborn")
    import seaborn as sns




class ControladorVideo:
    def __init__(self,enlace): 
        fb=Firebase('Interfaz/recetastextos/')
        self._idvideo = fb.reenumerar()
        self.enlacevideo=enlace
        self.yt=YouTube(self.enlacevideo)
        self.nombrevideo=''
        self.titulovideo=self.yt.title
        self.autorvideo=self.yt.author
        self.fechavideo=self.yt.publish_date
        self.duracionvideo=self.yt.length
        self.rec=RecursosAdicionales()
    """|DESCARGAR VIDEO URL: descarga el video de youtube
       |return: devuelve una ruta absoluta"""
    def descargarVideoURL(self):
        recetasVideos = 'recetasvideos/'
        #aqui creo un nuevo id para el nuevo video
        self._idvideo= self._idvideo+1
        #esta sera el archivo del video y su nuevo nombre
        nombre='receta'+str(self._idvideo)
        #le pedimos al pytube que solo nos descargue el audio y lo descargamos
        t=self.yt.streams.filter(file_extension='mp4').first().download(output_path=recetasVideos,filename=nombre+'.mp4')
        #devolvemos el nombre
        return nombre
    """|PARSEO VIDEO: pasa el video de .mp4 a .wav
       |nombre: es un string que se colocara el nombre del video
       |return: devuelve el nuevo nombre del audio en .wav"""
    def parseoVideo(self,nombre):
        recetasVideos = 'recetasvideos/'
        #tomamos el video en mp4 
        track = mp.VideoFileClip(recetasVideos+nombre+'.mp4')
        #cambiamos el video a .wav
        nombre_wav="{}.wav".format(nombre)
        track.audio.write_audiofile(recetasVideos+nombre_wav)
        track.close()
        return nombre
    """|SPEECH TEXT:Transforma el audio a texto
       |nombre: es un string que se colocara el nombre del video
       |return: devuelve un string con el texto devuelto"""
    def speech_text(self,nombre):
        recetasVideos = 'recetasvideos/'
        #instanciamos el recognizer
        r = sr.Recognizer()
        audio = sr.AudioFile(recetasVideos+nombre)
        with audio as source:
            audio_file = r.record(source)
        #transcribimos el audio a texto
        result = r.recognize_google(audio_file, language = 'es-ES')
        return result
    def data_json(self):
        return {"id":self._idvideo, "nombre":self.titulovideo, "autor": self.autorvideo, "fecha":str(self.fechavideo),"enlace":str(self.enlacevideo)}
    def indexar_datos(self):
        return self.rec.indexar_datos("Interfaz/recetastextos/indice.json",{"id":self._idvideo+1, "nombre":self.titulovideo, "autor": self.autorvideo, "fecha":str(self.fechavideo),"enlace":str(self.enlacevideo)})
    """|REPETIDO:Nos dice si el video ya se encuentra en nuestra bd
       |fileName: nombre del json
       |key: llave en donde queremos encontrar lo que buscamos
       |buscar: elemento que estamos buscando"""
    def repetido(self):
        return self.rec.buscar_json('Interfaz/recetastextos/indice.json','nombre',self.titulovideo)


class Depurador:
    
    def __init__(self): 
        self.rec=RecursosAdicionales()
    """|VIDEO: proceso etl donde extraemos al informacion del video 
       |enlace: es un string que se colocara el enlace del video"""
       
    def filtroDescarga(self, enlace_txtbox):
        if(re.search("\/playlist\?", enlace_txtbox)):
            self.lista(enlace_txtbox)
        else:
            self.video(enlace_txtbox)
    def video(self,enlace):
        try:
            #instanciamos el controlador de videos
            cv=ControladorVideo(enlace)
            fb=Firebase('Interfaz/recetastextos/')
            
            #paso 1: verificamos si existe en la database
            if fb.validar_database(cv.titulovideo)==False:
                #paso 2: guardamos en database datos principales
                
                #paso 3: descargamos el video
                cv.nombrevideo=cv.descargarVideoURL()
                print("id: "+str(cv._idvideo))
                fb.guardar_database(cv.data_json(),cv._idvideo)
                #paso 4: pasamos el video a .wav
                nombre=cv.parseoVideo(cv.nombrevideo)
                #paso 5: evaluamos los silencios 
                try:                
                    num_segm=self.rec.segcionarXsilencios(nombre)
                    result=""
                    for i in range(num_segm):
                        try:
                            result=result+str(cv.speech_text("../temp_audios/{}_extracto{}.wav".format(nombre,i+1)))
                            result=result+" "
                        except BaseException:
                            logging.exception("An exception was thrown!")
                            audio1=AudioSegment.from_wav("temp_audios/{}_extracto{}.wav".format(nombre,i+1))
                            duracion=audio1.duration_seconds
                            if duracion<=5:
                                print("El extracto {} es un silencio".format(i+1))
                            elif duracion<=180:
                                print("El extracto {} es música o ruido".format(i+1))
                            else:
                                print("Error importante en el extracto {}".format(i+1))
                    #paso 6: borramos los chunks temporales de audio
                    self.rec.eliminacion_audio("temp_audios","wav")
                    try:
                        quitarEmojis = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 'NULL')
                        tituloSinEmojis=cv.titulovideo.translate(quitarEmojis)
                        autorSinEmojis=cv.autorvideo.translate(quitarEmojis)
                        #paso 7: escribimos el texto recibido en un txt->se guarda en local
                        resultado=self.rec.escritura(cv.nombrevideo,"Titulo:"+tituloSinEmojis+"\n"+"Autor:"+autorSinEmojis+"\n"+"Fecha Publicacion:"+str(cv.fechavideo)+"\n"+"Enlace: "+str(cv.enlacevideo)+"\n"+"Entradilla:"+result)
                        #paso 8: guardamos el texto en una base de datos
                        fb.guardar_firebase(cv.nombrevideo+'.txt')
                        #paso 9: eliminamos los mp4
                        self.rec.eliminacion_audio("recetasvideos","mp4")
                    except BaseException:
                        logging.exception("An exception was thrown!")
                        print("No se ha podido eliminar los caracteres corruptos el video: "+ cv.nombrevideo + " - "+ cv.titulovideo)
                        self.rec.eliminacion_audio("recetasvideos","mp4")
                        return None   
                except BaseException:
                    logging.exception("An exception was thrown!")
                    print("No se ha podido transcribir el video: "+ cv.nombrevideo + " - "+ cv.titulovideo+" - "+cv.enlacevideo)
                    self.rec.eliminacion_audio("recetasvideos","mp4")
                    self.rec.eliminacion_audio("temp_audios","wav")
                    return None
            else:
                print('Este video se encuentra en la base de datos.')
                resultado=""
            return resultado
        except BaseException:
            logging.exception("An exception was thrown!")
            print("No se ha podido descargar el video: "+ cv.nombrevideo + " - "+ cv.titulovideo)
            return None
    def lista(self, enlace):
        playlist_urls = Playlist(enlace)
        for url in playlist_urls:
            self.video(url)


class RecursosAdicionales:
    """|ESCRITURA: escribe textos txt
       |nombre: nombre del 
       |return: devuelve el audio en texto"""    
    def escritura(self,nombre,texto):
        recetasTextos = './Interfaz/recetastextos/'
        if not(os.path.exists(recetasTextos)):
            os.mkdir(recetasTextos)
        f = open(recetasTextos+nombre+'.txt', 'w')
        f.write(texto)
        f = open(recetasTextos+nombre+'.txt', "r")
        print(f.read())
        f.close()
        
    def lectura_json(self,fileName):
        if self.documento_vacio(fileName):
            with open(fileName, "r") as file:
                    archivo=json.load(file)
        else: 
            archivo=[]
            print('El documento se encuentra vacio.')
        return archivo
    
    def escritura_json(self,fileName,data):
        with open(fileName, "w") as file:
                json.dump(data, file)
                file.close()
    def buscar_json(self,fileName,key,buscar):
        encontrado=False
        if self.documento_vacio(fileName):
            archivo_json=self.lectura_json(fileName)
            for item in archivo_json:
                if buscar in item[key]:
                    print('encontrado')
                    encontrado=True
                    #no me gusta usar esto pero no tengo idea de como usar un while con json
                    break
        return encontrado
    def documento_vacio(self,fileName):
        return os.stat(fileName).st_size != 0
    def indexar_datos(self,fileName,adicion):
        if not(os.path.exists(fileName)):
            os.mkdir(fileName)
        data=[]
        data=self.lectura_json(fileName)
        data.append(adicion)
        self.escritura_json(fileName,data)
        
    def eliminacion_audio(self,path,tipo):
        url = './'+path+'/'
        py_files = glob.glob(url+'*.'+tipo)
        for py_file in py_files:
            try:
                os.remove(py_file)
            except OSError as e:
                print(f"Error:{ e.strerror}")
    
    def segcionarXsilencios(self,audio):
        audio1=AudioSegment.from_wav("./recetasvideos/"+audio+".wav")
        var_min=1900
        salir=False
        while salir==False:
            samples = audio1.get_array_of_samples()
            segundo=88521
            index=[]
            for i in range(0,len(samples),int(segundo/5)):
                dataSeg = samples[i:int(segundo/5)+i]
                media=np.mean(dataSeg)
                var=np.var(dataSeg)
                if -10<=media<=10 and var<=var_min:
                    index.append(i)

            borrar=[]
            guardado=0
            for i in range(len(index)-1):
                if index[i+1]<=index[i]+(20*segundo):
                    if i==0:
                        tiempo=(index[i])/segundo
                    else:
                        tiempo=(index[i+1]-guardado)/segundo
                    if tiempo<=120:
                        borrar.append(i)
                    else:
                        guardado=index[i]
                else:
                    guardado=index[i]

            final=np.delete(index, borrar, axis=0) 
            extractos=[]
            if len(final)==0:
                var_min=var_min*10
                salir=False
            else:
                for i in range(len(final)):
                    if i==0:
                        extractos.append(samples[:final[i]])
                    else:
                        extractos.append(samples[final[i-1]:final[i]])
                extractos.append(samples[final[i]:])
                salir=True

        for i in range(len(extractos)):
            nombre=""
            new_sound = audio1._spawn(extractos[i])
            nombre="temp_audios/{}_extracto{}.wav".format(audio,i+1)
            new_sound.export(nombre,format="wav")
        #print(len(extractos))
        return len(extractos)





class Firebase:
    def __init__(self,ubicacion):
        self.ubi=ubicacion
        
        self.config={"apiKey": "AIzaSyDDg9WOlFJxnEJoxomYtsnkJfsI4TgoL_E","authDomain": "eateaser-741d4.firebaseapp.com","databaseURL" : "https://eateaser-741d4-default-rtdb.firebaseio.com/","projectId": "eateaser-741d4","storageBucket": "eateaser-741d4.appspot.com","messagingSenderId": "706351391410","appId": "1:706351391410:web:6abc2cabd6bf83843b5fab","measurementId": "G-YZZCBRHNBT"};
        self.firebase=self.conexion_firebase()
        self.database=self.firebase.database()
    def conexion_firebase(self):
        return pyrebase.initialize_app(self.config)
    def guardar_firebase(self,nom):
        storage=self.firebase.storage()
        storage.child(self.ubi+nom).put(self.ubi+nom)
    def eliminar_firebase(self,nom):
        self.firebase.storage().delete(self.ubi+nom)
    def guardar_database(self,data,_id):
        self.database.child('Recetas').child(_id).set(data)
    def validar_database(self,data):
        validar=self.database.get()
        encontrado=False
        for a in validar.each():
            if  data in str(a.val()):
                encontrado=True
                #no me gusta usar esto pero no tengo idea de como usar un while con json
                break
        return encontrado
    def reenumerar(self):
        recetas=self.database.child("Recetas").get()
        id=0
        for item in recetas.each():
            id=item.key()
        return int(id)



class ProcesarDocumentos:     
    def lectura(self):
        procDoc=ProcesarDocumentos()
        rutaCarpetasPorCategoria = "./recetastextos/"
        listaCarpetasFinal = []
        #estos string nos servirán para guardar todos los textos de los txt por cada una de las carpetas
        carpetaArroz = carpetaBebidas = carpetaCarnes = carpetaMarisco = carpetaPasta = carpetaPescados = carpetaPlatosMenores = carpetaVerduras = ''
        #sacamos una lista de todas las carpetas
        listaCarpetas = os.listdir(rutaCarpetasPorCategoria)
        #print(listaCarpetas)
        #print(len(listaCarpetas))
        #recorremos todas las carpetas
        i=0
        for lc in listaCarpetas:
            #cogemos el nombre de la carpeta y se lo concatenamos a la ruta anterior
            rutaPorCarpeta = rutaCarpetasPorCategoria + lc + '/'
            print(str(i)+rutaPorCarpeta+'-----------------')
            if(i==0):
                carpetaArroz = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                #print(carpetaArroz)
                listaCarpetasFinal.append(carpetaArroz)
            if(i==1):
                carpetaBebidas = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaBebidas)
            if(i==2):
                carpetaCarnes = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaCarnes)
            if(i==3):
                carpetaMarisco = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaMarisco)
            if(i==4):
                carpetaPasta = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaPasta)
            if(i==5):
                carpetaPescados = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaPescados)
            if(i==6):
                carpetaPlatosMenores = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaPlatosMenores)
            if(i==7):
                carpetaVerduras = procDoc.resultadoStringCarpeta(rutaPorCarpeta)
                listaCarpetasFinal.append(carpetaVerduras)
            i=i+1
        return listaCarpetasFinal
    def lecturaTesting(self):
        procDoc=ProcesarDocumentos()
        rutaCarpetaTesting = "./Interfaz/Carpeta Testing/"
        carpetaTesting = procDoc.resultadoStringCarpeta(rutaCarpetaTesting)
        return carpetaTesting
    def resultadoStringCarpeta(self, rutaPorCarpeta):
        strCarpeta=[]
        #vemos el contenido de la carpeta en la que estamos iterando
        listaTxt = os.listdir(rutaPorCarpeta)
        print(listaTxt)
        #recorremos todos los archivos de la carpeta
        for lt in listaTxt:
            #concatenamos la ruta de la carpeta con el nombre de los archivos que contiene esta
            rutaTxt = rutaPorCarpeta + lt
            #al ir iterando pasaremos por todos los archivos modificando la variable de la ruta para poder hacer un open con ella
            #file = open(filename, encoding="utf8")
            try:
                with open(rutaTxt, 'r') as f: 
                    #al hacer el open leemos lo que hay dentro del archivo con f.read(), y esto lo guardamos dentro de un string inicializado al inicio del todo
                    strCarpeta.append(f.read())
            except:
                with open(rutaTxt, 'r',encoding="utf8") as f: 
                    #al hacer el open leemos lo que hay dentro del archivo con f.read(), y esto lo guardamos dentro de un string inicializado al inicio del todo
                    strCarpeta.append(f.read())
                
        return strCarpeta
    def leer_stopwords(self, path):
        with open(path) as f:
            # Lee las stopwords del archivo y las guarda en una lista
            mis_stopwords = [line.strip() for line in f]
        return mis_stopwords
    def tratamientoTextos(self, info):
        #Eliminamos posibles horas del titulo
        textoSinSimbolos = re.sub("\d+:\d+:\d+", "" , info)
        #Eliminamos posibles fechas
        textoSinSimbolos = re.sub("\d+-\d+-\d+", "" , textoSinSimbolos)
        #Eliminamos todos los fin de enlace
        textoSinSimbolos = re.sub("v=.*", "" , textoSinSimbolos)
        #Eliminamos todos los simbolos del texto (,.;:?¿!!) etc
        textoSinSimbolos = re.sub("[^0-9A-Za-z_]", " " , textoSinSimbolos)
        #Sacamos todos los tokens del texto y los metemos en una lista
        textoTokenizado = nltk.tokenize.word_tokenize(textoSinSimbolos)
        #una lista no tiene lower asique pasamos el lower con map a toda la lista
        textoMinusculas = (map(lambda x: x.lower(), textoTokenizado))
        #Le pasa un stopword de palabras en español a la lista de palabras que le llega
        #stop_words_sp = set(stopwords.words('spanish'))
        stop_words_sp = self.leer_stopwords("./rapidminer/stop_words_spanish.txt")
        pasarStopWords = [i for i in textoMinusculas if i not in stop_words_sp]
        #Aplicamos la normalizacion mediante stemming
        #SnowStem = nltk.SnowballStemmer(language = 'spanish')
        # Crear un objeto SnowballStemmer para el idioma español
        stemmer = RSLPStemmer()
        listaStems = [stemmer.stem(word) for word in pasarStopWords]
        return listaStems



class modelos:
    def __init__(self):
        self.preprocesamiento()
     
    def preprocesamiento(self):
        
        #Se crea la función que vectoriza los array de las recetas (calcula la frecuencia de las palabras) lo que
        #convierte una lista de palabras en un array de frecuencias
        self.vectorizer = CountVectorizer(analyzer = "word",  tokenizer = None, preprocessor = None,  stop_words = None,  max_features = 10000) 

        #Se separa el set de datos en datos de entrenamiento y de testeo. En este caso se divide en 80%-20%
        #Creandose 4 variables -> 
        #X_train: Conjunto de recetas de entrenamiento (X_cv: en el testeo) 
        #Y_train: clasificación de las recetas en los datos de entrenamiento (Y_cv: en el testeo)    
        self.X_train, self.X_cv, self.Y_train, self.Y_cv = train_test_split(df["receta"], df["clasif"], test_size = 0.2, random_state=42)
        self.Y_train=list(self.Y_train)

        #Ahora vecrtorizamos X_train y X_cv para poder meterlo en el modelo de clasificación
        #Set de entrenamiento
        arrayTemp=[]
        for i,j in enumerate(self.X_train):           #El fit_transform funciona con string de frases enteras y automáticamente tokeniza las palabras por lo que hayq ue volver a juntar las palabras en una frase
            arrayTemp.append(" ".join(j))
        self.X_train = self.vectorizer.fit_transform(arrayTemp)
        self.X_train = self.X_train.toarray()

        #Set de testeo
        arrayTemp=[]
        for i,j in enumerate(self.X_cv):
            arrayTemp.append(" ".join(j))
        self.X_cv = self.vectorizer.transform(arrayTemp)
        self.X_cv = self.X_cv.toarray()
        self.Y_train=list(self.Y_train)
        self.Y_cv=list(self.Y_cv)
        
    def Entrenar_RF(self):
        # Grid de hiperparámetros evaluados
        # ==============================================================================
        
        print(self.X_train.shape)
        param_grid = ParameterGrid(
                        {'n_estimators': [1000],
                         'max_features': [5, 7, 9],
                         'max_depth'   : [None, 3, 10, 20],
                         'criterion'   : ['gini', 'entropy']
                        }
                    )

        # Loop para ajustar un modelo con cada combinación de hiperparámetros
        # ==============================================================================
        resultados = {'params': [], 'oob_accuracy': []}

        for params in param_grid:

            modelo = RandomForestClassifier(
                        oob_score    = True,
                        n_jobs       = -1,
                        random_state = 123,
                        ** params
                     )

            modelo.fit(self.X_train, self.Y_train)

            resultados['params'].append(params)
            resultados['oob_accuracy'].append(modelo.oob_score_)
            print(f"Modelo: {params} \u2713")

        # Resultados
        # ==============================================================================
        resultados = pd.DataFrame(resultados)
        resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
        resultados = resultados.sort_values('oob_accuracy', ascending=False)
        resultados = resultados.drop(columns = 'params')
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
                      'max_depth'   : [None, 3, 10, 20],
                      'criterion'   : ['gini', 'entropy']
                     }

        # Búsqueda por grid search con validación cruzada
        # ==============================================================================
        grid = GridSearchCV(
                estimator  = RandomForestClassifier(random_state = 123),
                param_grid = param_grid,
                scoring    = 'accuracy',
                n_jobs     = multiprocessing.cpu_count() - 1,
                cv         = RepeatedKFold(n_splits=5, n_repeats=3, random_state=123), 
                refit      = True,
                verbose    = 0,
                return_train_score = True
               )

        grid.fit(X = self.X_train, y = self.Y_train)

        # Resultados
        # ==============================================================================
        resultados = pd.DataFrame(grid.cv_results_)
        resultados.filter(regex = '(param*|mean_t|std_t)') \
            .drop(columns = 'params') \
            .sort_values('mean_test_score', ascending = False) \
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
        
        
    def predecir_RF(self,txt):
        
        self.pred = self.vectorizer.transform(txt)
        self.pred = self.pred.toarray()
        predictions = self.forest.predict(self.pred) 
        #print("resultado: " , predictions)
        return list(predictions)
    def predecir_Carpeta(self,txt):
        
        p=ProcesarDocumentos()
        carpeta=p.resultadoStringCarpeta(txt)

        resultados=[]
        for i in range(len(carpeta)):
            text=p.tratamientoTextos(carpeta[i])
            hey=[" ".join(text)]
            resultados.append(self.predecir_RF(hey))
        #print("Resultados: {}".format(resultados))
        
        resultados=[]
        for i in range(len(carpeta)):
            text=p.tratamientoTextos(carpeta[i])
            hey=" ".join(text)
            resultados.append(hey)
        self.pred1 = self.vectorizer.transform(resultados)
        self.pred1 = self.pred1.toarray()
        predictions = self.forest.predict(self.pred1) 
        #print("resultado: " , predictions)
        return predictions
        
    def Entrenar_KNN(self):  
        #self.preprocesamiento()
        

        vecinos = KNeighborsClassifier() 
        vecinos = vecinos.fit(self.X_train, self.Y_train)

        predictions = vecinos.predict(self.X_cv) 
        self.Y_cv=list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))

    
    def Entrenar_SVM(self):  
        #self.preprocesamiento()
        
        #Create a svm Classifier
        clf = svm.SVC(kernel='linear') # Linear Kernel

        #Train the model using the training sets
        clf.fit(self.X_train, self.Y_train)
        

        predictions = clf.predict(self.X_cv) 
        self.Y_cv=list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))
        
        
        cv=cross_val_score(clf, self.X_train, self.Y_train, cv=10)
        
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

    def Entrenar_Bayes(self):  
        
        #Create a svm Classifier
        gaus = GaussianNB() # Linear Kernel

        #Train the model using the training sets
        gaus.fit(self.X_train, self.Y_train)
        

        predictions = gaus.predict(self.X_cv) 
        self.Y_cv=list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))
        
        
        cv=cross_val_score(gaus, self.X_train, self.Y_train, cv=10)
        
        print("CV -> {}".format(cv))   
        
    
    def regresionMultinomial(self):
        
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(self.X_train, self.Y_train)
        

        predictions = model.predict(self.X_cv) 
        self.Y_cv=list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))
        
        
        cv=cross_val_score(model, self.X_train, self.Y_train, cv=10)
        
        print("CV -> {}".format(cv)) 
    def Entrenar_RedNeuronal(self):
        
        xtrain=[]
        for i in range(len(self.X_train)):     
            xtrain.append(list(self.X_train[i]))
        #xtrain=np.array(xtrain)

        xcv=[]
        for i in range(len(self.X_cv)):     
            xcv.append(list(self.X_cv[i]))
        #xcv=np.array(xcv)
        #Y_train=np.array(Y_train)
        #Y_cv=np.array(Y_cv)


        print(type(xtrain))
        print(type(self.Y_train))
        print(type(xcv))
        print(type(self.Y_cv))
        
        
        clear_session()
        

        #input_dim = xtrain.shape[1] #.shape[0]  # Number of features
        input_dim= len(xtrain[0])
        model = Sequential()
        model.add(layers.Dense(7, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        '''
        model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])
        
        '''
        
        loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
        model.compile(loss=loss_fn, 
                       optimizer='adam', 
                       metrics=['accuracy'])
        
        print(model.summary())

        xtrain2=[]
        for i in range(len(xtrain)):
            temp=[]
            for j in range(len(xtrain[i])):
                temp.append(int(xtrain[i][j]))
            xtrain2.append(temp)
        
        xcv2=[]
        for i in range(len(xcv)):
            temp=[]
            for j in range(len(xcv[i])):
                temp.append(int(xcv[i][j]))
            xcv2.append(temp)
            
        ytrain2=[]
        for i in range(len(self.Y_train)):
            ytrain2.append(int(self.Y_train[i]))
            
        ycv2=[]
        for i in range(len(self.Y_cv)):
            ycv2.append(int(self.Y_cv[i]))
            
        print(type(xtrain2[0][0]))    
        self.xtrain2=xtrain2        
        self.xtrain=xtrain
        '''
        history = model.fit(xtrain2, self.Y_train,
                     epochs=10,
                     verbose=False,
                     validation_data=(xcv, self.Y_cv),
                     batch_size=10)
        '''
        history = model.fit(xtrain2, ytrain2,
                     epochs=10,
                     verbose=False,
                     validation_data=(xcv2, ycv2),
                     batch_size=10)

        
        clear_session()
        '''
        loss, accuracy = model.evaluate(xtrain, self.Y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(xcv, self.Y_cv, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        '''
        loss, accuracy = model.evaluate(xtrain2, ytrain2, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(xcv2, ycv2, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))



class modelosTFIDF:
    def __init__(self):
        self.tfidf()
    
    def tfidf(self):
        hola=[]
        for i,j in enumerate(df['receta']):
            hola.append(" ".join(j))
        self.vectorizers= TfidfVectorizer(max_features=4000)    
        self.vect = self.vectorizers.fit_transform(hola)
        arr=self.vect.toarray()
        variable=self.vectorizers.get_feature_names()
        

        variables=dict.fromkeys(variable,None)

        tf1=pd.DataFrame(variables,index=[0])
        for i in range(len(df['clasif'])):
            tf1.loc[i]=arr[i]
        
        self.X_train, self.X_cv, self.Y_train, self.Y_cv = train_test_split(tf1, df['clasif'], test_size = 0.2, random_state=42)
        self.Y_train=list(self.Y_train)
        self.Y_cv=list(self.Y_cv)
         
  
        
    def Entrenar_RF(self):
        # Grid de hiperparámetros evaluados
        # ==============================================================================
        
        print(self.X_train.shape)
        param_grid = ParameterGrid(
                        {'n_estimators': [1000],
                         'max_features': [5, 7, 9],
                         'max_depth'   : [None, 3, 10, 20],
                         'criterion'   : ['gini', 'entropy']
                        }
                    )

        # Loop para ajustar un modelo con cada combinación de hiperparámetros
        # ==============================================================================
        resultados = {'params': [], 'oob_accuracy': []}

        for params in param_grid:

            modelo = RandomForestClassifier(
                        oob_score    = True,
                        n_jobs       = -1,
                        random_state = 123,
                        ** params
                     )

            modelo.fit(self.X_train, self.Y_train)

            resultados['params'].append(params)
            resultados['oob_accuracy'].append(modelo.oob_score_)
            print(f"Modelo: {params} \u2713")

        # Resultados
        # ==============================================================================
        resultados = pd.DataFrame(resultados)
        resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
        resultados = resultados.sort_values('oob_accuracy', ascending=False)
        resultados = resultados.drop(columns = 'params')
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
                      'max_depth'   : [None, 3, 10, 20],
                      'criterion'   : ['gini', 'entropy']
                     }

        # Búsqueda por grid search con validación cruzada
        # ==============================================================================
        
        grid = GridSearchCV(
                estimator  = RandomForestClassifier(random_state = 123),
                param_grid = param_grid,
                scoring    = 'accuracy',
                n_jobs     = multiprocessing.cpu_count() - 1,
                cv         = RepeatedKFold(n_splits=5, n_repeats=3, random_state=123), 
                refit      = True,
                verbose    = 0,
                return_train_score = True
               )

        grid.fit(X = self.X_train, y = self.Y_train)

        # Resultados
        # ==============================================================================
        resultados = pd.DataFrame(grid.cv_results_)
        resultados.filter(regex = '(param*|mean_t|std_t)') \
            .drop(columns = 'params') \
            .sort_values('mean_test_score', ascending = False) \
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
        #self.preprocesamiento()
        
        #Create a svm Classifier
        m_SVM = svm.SVC(kernel='linear') # Linear Kernel

        #Train the model using the training sets
        m_SVM.fit(self.X_train, self.Y_train)
        

        predictions = m_SVM.predict(self.X_cv) 
        self.Y_cv=list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))
        
        
        cv=cross_val_score(m_SVM, self.X_train, self.Y_train, cv=10)
        self.m_SVM=m_SVM
        
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
        
        
        #Create a svm Classifier
        gaus = GaussianNB() # Linear Kernel

        #Train the model using the training sets
        gaus.fit(self.X_train, self.Y_train)
        

        predictions = gaus.predict(self.X_cv) 
        self.Y_cv=list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))
        
        
        cv=cross_val_score(gaus, self.X_train, self.Y_train, cv=10)
        self.gaus=gaus
        print("CV -> {}".format(cv))   
        
    
    def regresionMultinomial(self):
        
        M_mult = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        M_mult.fit(self.X_train, self.Y_train)
        
        predictions = M_mult.predict(self.X_cv) 
        self.Y_cv=list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))
        
        #cv=cross_val_score(M_mult, self.X_train, self.Y_train, cv=10)
        self.M_mult=M_mult
        #print("CV -> {}".format(cv)) 
    
    def predecir_RF(self,txt):
        
        self.pred = self.vectorizers.transform(txt)
        self.pred = self.pred.toarray()
        predictions = self.M_mult.predict(self.pred) 
        print("resultado: " , predictions)
        
    def clasificar(self,modelo,txt):
        
        self.pred = self.vectorizers.transform(txt)
        self.pred = self.pred.toarray()
        predictions = modelo.predict(self.pred) 
        print("resultado: " , predictions)
    
    def predecir_Carpeta(self,txt):
        
        p=ProcesarDocumentos()
        carpeta=p.resultadoStringCarpeta(txt)

        resultados=[]
        for i in range(len(carpeta)):
            text=p.tratamientoTextos(carpeta[i])
            hey=[" ".join(text)]
            resultados.append(self.predecir_RF(hey))
        #print("Resultados: {}".format(resultados))
        
        resultados=[]
        for i in range(len(carpeta)):
            text=p.tratamientoTextos(carpeta[i])
            hey=" ".join(text)
            resultados.append(hey)
        self.pred1 = self.vectorizers.transform(resultados)
        self.pred1 = self.pred1.toarray()
        predictions = self.M_mult.predict(self.pred1) 
        #print("resultado: " , predictions)
        return predictions
        
    def Entrenar_KNN(self):  
        #self.preprocesamiento()
        

        vecinos = KNeighborsClassifier() 
        vecinos = vecinos.fit(self.X_train, self.Y_train)

        predictions = vecinos.predict(self.X_cv) 
        self.Y_cv=list(self.Y_cv)
        print("Accuracy: ", accuracy_score(self.Y_cv, predictions))
        
    
    def Entrenar_RedNeuronal(self):
        xtrain=[]
        for i in range(len(self.X_train)):     
            xtrain.append(list(self.X_train.iloc[i]))
        #xtrain=np.array(xtrain)

        xcv=[]
        for i in range(len(self.X_cv)):     
            xcv.append(list(self.X_cv.iloc[i]))
        #xcv=np.array(xcv)
        #Y_train=np.array(Y_train)
        #Y_cv=np.array(Y_cv)


        print(type(xtrain))
        print(type(self.Y_train))
        print(type(xcv))
        print(type(self.Y_cv))
        
        
        clear_session()
        

        #input_dim = xtrain.shape[1] #.shape[0]  # Number of features
        input_dim= len(xtrain[0])
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

    def guardarModelo(self,modelo,nombre):
        
        joblib.dump(modelo, './Interfaz/modelos/{}.pkl'.format(nombre)) # Guardo el modelo.
    
    def cargarModelo(self,nombre):
        
        return joblib.load('./Interfaz/modelos/{}.pkl'.format(nombre))



class Index(QtWidgets.QMainWindow):
    def __init__(self):
        super(Index, self).__init__()
        # Cargamos el .ui file
        uic.loadUi('index.ui', self)
        #cargamos widgets
        self.setWidgets()
        #activamos botones
        self.activarBotones()
    #Aqui se declaran los widgets
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
        self.personalizar_boton(self.rober, 'rober.jpeg')
        self.acceder.setVisible(False)
        self.icono.setVisible(False)
        self.apagar_widgets(False)
    #aqui ponemos los eventos de los botones
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
        self.acceder.clicked.connect(self.openTrain)
    def menuClicked(self, titulo, descripcion, acceder, widgets,dir):
        self.titulo.setText(titulo)
        self.descripcion.setText(descripcion)
        self.acceder.setVisible(acceder)
        self.apagar_widgets(widgets)
        self.state = titulo
        if(widgets==False):
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
    def openTrain(self):
        if self.state=='Fase de Entrenamiento':
            self.gui = Train()
            self.gui.show()
            self.gui.showMaximized()
            self.close()
        if self.state=='Fase de Testeo':
            self.gui = Test()
            self.gui.show()
            self.gui.showMaximized()
            self.close()
        if self.state=='Aplicación':
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.gui = App()
            self.gui.show()
            self.gui.showMaximized()
            QApplication.restoreOverrideCursor()
            self.close()
class Train(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eat Easer Train page")
        self.setWindowIcon(QIcon("imagenes/EatEaser-Logo.png"))
        #variables globales
        self.setWidgets()
        self.setLayouts()
        self.stylesheet()
        self.addlayout_to_layouts()
        self.addwindgets_to_layouts()
        self.activarBotones()
        self.opcionModeloElegida=0

    def setLayouts(self):
        # layout grande
        self.layout = QGridLayout()
        # partes del layout grande
        self.izqlayout = QGridLayout()
        self.derlayout = QGridLayout()
        # grid de la ruta
        self.rutalayout = QGridLayout()
        # grid de algoritmos
        self.algoritmolayout = QGridLayout()
        # grjd guardar
        self.guardar = QGridLayout()
        self.seleccionlayout = QHBoxLayout()
        self.grafico = QVBoxLayout()
        self.setLayout(self.layout)


    def setWidgets(self):
        self.seleccionados = []
        self.checkboxes = []
        self.varableSeleccionCarpetaGuardarModelo = ""
        # labels de ruta
        self.lcategoria = QLabel('Selecciona la categoria')
        # combobox de ruta
        self.cbcategoria = QComboBox()
        self.cbcategoria.setFixedSize(800, 40)
        self.cbcategoria.addItems(os.listdir('recetastextos/'))
        # zona derecha del layout labels
        self.linfo = QPushButton()
        self.ltitulo = QLabel('Nombre Algoritmo')
        self.ldescrip = QLabel('Descripcion Algoritmo')
        self.vista = QLabel('Vista Algoritmo')
        self.fondo = QLabel()
        # estilizar labels
        self.linfo.setIcon(QIcon('imagenes/informacion.png'))
        self.linfo.setFixedSize(QtCore.QSize(400, 80))
        size = QSize(50, 50)
        self.linfo.setIconSize(size)
        # labels algoritmo
        self.lalgoritmo = QLabel('Algoritmo:')
        # botones de algorimos
        self.btn_svm = QPushButton('SVM')
        self.btn_rf = QPushButton('Random-Forest')
        self.btn_mr = QPushButton('Multinomial Regression')
        self.btnalgoritmo = QPushButton()
        # estilizamos botones
        self.btn_svm.setFixedSize(QtCore.QSize(400, 80))
        self.btn_rf.setFixedSize(QtCore.QSize(400, 80))
        self.btn_mr.setFixedSize(QtCore.QSize(400, 80))
        # grid del grafico
        self.layout13 = QLabel('d')
        size = QSize(50, 50)
        self.btnalgoritmo.setIconSize(size)
        self.btnalgoritmo.setIcon(QIcon('imagenes/boton-de-play.png'))
        self.btnalgoritmo.setFixedSize(QtCore.QSize(80, 80))
        # botones de ruta
        self.anadir = QPushButton()
        self.anadir.setIcon(QIcon('imagenes/cargar.png'))
        self.anadir.setFixedSize(QtCore.QSize(40, 40))
        self.nuevo = QPushButton()
        self.nuevo.setIcon(QIcon('imagenes/add.png'))
        self.nuevo.setFixedSize(QtCore.QSize(40, 40))
        # botones de grid guardar
        self.borrar = QPushButton()
        self.path_btn = QPushButton('')
        self.btn_guardar = QPushButton()
        # estilizamos botones
        self.borrar.setFixedSize(QtCore.QSize(80, 80))
        self.borrar.setIcon(QIcon('imagenes/delete.png'))
        self.path_btn.setIcon(QIcon('imagenes/lupa.png'))
        self.btn_guardar.setIcon(QIcon('imagenes/guardar-el-archivo.png'))
        # form del grid guardar
        self.formguardar = QLineEdit()
        self.lform = QLabel("Guardar modelo:")
        # boton de retorno izquierdo
        self.retorno = QPushButton(u"\u2190" + ' Main Page/ Entrenamiento')

    def stylesheet(self):
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
        # estilizamos los botones
        self.btnalgoritmo.setStyleSheet(sbotones)
        self.btn_svm.setStyleSheet(sbtnalgoritmo)
        self.btn_rf.setStyleSheet(sbtnalgoritmo)
        self.btn_mr.setStyleSheet(sbtnalgoritmo)
        self.nuevo.setStyleSheet(sbtnruta)
        self.anadir.setStyleSheet(sbtnruta)
        self.cbcategoria.setStyleSheet(scbcategoria)
        self.lcategoria.setStyleSheet(scategoria)
        # estilizamos zona derecha
        self.ltitulo.setStyleSheet(stextos_derecha)
        self.ldescrip.setStyleSheet(stextos_derecha)
        self.vista.setStyleSheet(stextos_derecha)
        self.fondo.setStyleSheet(sinfo)
        self.linfo.setStyleSheet(sinfo)
        self.lalgoritmo.setStyleSheet(salgoritmo)
        self.retorno.setStyleSheet(sretorno)
        self.btn_guardar.setStyleSheet(sbotones)
        self.path_btn.setStyleSheet(sbotones)
        self.lform.setStyleSheet(sform)
        self.layout13.setStyleSheet('background-color:white')
    def addlayout_to_layouts(self):
        # aniadimos los layouts al total
        self.layout.addLayout(self.izqlayout, 0, 0)
        self.layout.addLayout(self.derlayout, 0, 1)
        self.layout.setColumnStretch(0, 3)
        self.layout.setColumnStretch(1, 1)
        # aniadimos los layouts al lado izq
        self.izqlayout.addLayout(self.rutalayout, 1, 0)
        self.izqlayout.addLayout(self.algoritmolayout, 2, 0)
        self.izqlayout.addLayout(self.seleccionlayout, 3, 0)
        self.izqlayout.addLayout(self.grafico, 4, 0)
        self.izqlayout.addLayout(self.guardar, 5, 0)
        # estilizamos los layouts
        self.izqlayout.setRowStretch(0, 1)
        self.izqlayout.setRowStretch(1, 1)
        self.izqlayout.setRowStretch(2, 1)
        self.izqlayout.setRowStretch(3, 1)
        self.izqlayout.setRowStretch(4, 2)
        self.izqlayout.setRowStretch(5, 1)
    def addwindgets_to_layouts(self):
        # aniadimos al layout de ruta
        self.rutalayout.addWidget(self.lcategoria, 0, 0, 1, 1)
        self.rutalayout.addWidget(self.cbcategoria, 0, 1, 1, 1)
        self.rutalayout.addWidget(self.anadir, 0, 2, 1, 1)
        self.rutalayout.addWidget(self.nuevo, 0, 3, 1, 1)
        # aniadimos widgets en lado derecho
        self.derlayout.addWidget(self.fondo, 0, 0, 6, 1)
        self.derlayout.addWidget(self.linfo, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(self.ltitulo, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(self.ldescrip, 2, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(self.vista, 3, 0, 3, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.rowStretch(1)
        # aniadimos los widgets a guardar
        self.guardar.addWidget(self.lform, 0, 0, 1, 1)
        self.guardar.addWidget(self.formguardar, 0, 1, 1, 1)
        self.guardar.addWidget(self.btn_guardar, 0, 2)
        self.guardar.addWidget(self.path_btn, 0, 3)
        # aniadimos al grid de algoritmos
        self.algoritmolayout.addWidget(self.lalgoritmo, 0, 0, 1, 4)
        self.algoritmolayout.addWidget(self.btn_svm, 1, 0, 2, 1)
        self.algoritmolayout.addWidget(self.btn_rf, 1, 1, 2, 1)
        self.algoritmolayout.addWidget(self.btn_mr, 1, 2, 2, 1)
        self.algoritmolayout.addWidget(self.btnalgoritmo, 1, 3, 2, 1)
        # aniadimos a la seleccion
        self.seleccionlayout.addWidget(self.borrar, 1, QtCore.Qt.AlignLeft)
        self.grafico.addWidget(self.layout13)
        self.izqlayout.addWidget(self.retorno, 0, 0)
    def activarBotones(self):
        # eventos de botones
        self.anadir.clicked.connect(self.aniadir_boton)
        self.borrar.clicked.connect(self.eliminar_boton)
        self.path_btn.clicked.connect(self.aniadir_directorio)
        self.btn_guardar.clicked.connect(self.guardarModelo)
        self.btn_svm.clicked.connect(
            lambda: (self.informacion('Algoritmo SVM', 'Este algoritmo hace esto y esto y esto'), 
                    self.modeloEntrenamientoElegido(1)))
        self.btn_mr.clicked.connect(
            lambda: (self.informacion('Algoritmo Multinomial Regression', 'Este algoritmo hace esto y esto y esto'),
                    self.modeloEntrenamientoElegido(3)))
        self.btn_rf.clicked.connect(
            lambda: (self.informacion('Algoritmo Random Forest', 'Este algoritmo hace esto y esto y esto'),
                    self.modeloEntrenamientoElegido(2)))
        self.btnalgoritmo.clicked.connect(self.vista_previa)
        self.nuevo.clicked.connect(self.aniadir_categoria)
        self.retorno.clicked.connect(self.volver)
    def informacion(self,titulo,descripcion):
            self.ltitulo.setText(titulo)
            self.ldescrip.setText(descripcion)
    
    def modeloEntrenamientoElegido(self, opcionElegida):
        self.opcionModeloElegida = opcionElegida
    
    #def cargarModeloSeleccionadoUser(self):
    #    modeloTfIdf = modelosTFIDF()
    #    if(self.opcionModeloElegida==1):
    #        modeloTfIdf
    #    elif(self.opcionModeloElegida==2):
    #        modeloTfIdf.
    #    elif(self.opcionModeloElegida==3):
    #        modeloTfIdf
        
        
    def vista_previa(self):
        self.vista.setText('')
        i = 0
        self.total_archivos=0

        carpetas=''
        #verificamos si hay seleccionados

        if len(self.seleccionados)==0 or self.ltitulo.text()=='Nombre Algoritmo' :

            self.mensaje_error('Campos vacios.')
        else:
            #self.cargarModeloSeleccionadoUser()
            #verificamos si hay algoritmo seleccionado
            for i in self.seleccionados:

                size = len(os.listdir('recetastextos/' + i))

                self.total_archivos = size + self.total_archivos
                texto=i + ': ' + str(size) + ' archivos\n'

                carpetas=carpetas+'\n'+texto



        # le añado todos los que esten en listbox
        self.vista.setText(carpetas+'\n'+'TOTAL: ' + ': ' + str(self.total_archivos) + ' archivos\n')

    def mensaje_error(self,mensaje):
        QMessageBox.critical(
            self,
            "Error",
            mensaje,
            buttons=QMessageBox.Discard | QMessageBox.NoToAll | QMessageBox.Ignore,
            defaultButton=QMessageBox.Discard,
        )

    def aniadir_boton(self):
        self.add=QCheckBox(self.cbcategoria.currentText())
        self.seleccionados.append(str(self.cbcategoria.currentText()))
        self.add.setStyleSheet('QPushButton{'+'background-color:transparent;border-radius:12px;border:2px solid black;font-family:"Bahnschrift Light";font-size:20px;letter-spacing:3px;}QPushButton:hover{background-color:'
                                              'black;color:white}')

        self.seleccionlayout.addWidget(self.add)
        self.checkboxes.append(self.add)
        print(self.seleccionados)

    def eliminar_boton(self):

        i=0
        for c in self.checkboxes:
            if c.isChecked()==True:

                c.deleteLater()
                self.checkboxes.pop(i)
                self.seleccionados.pop(i)
                print(c.text())

            i=i+1

    def aniadir_categoria(self):
        r=QFileDialog.getExistingDirectory(self, "Select Directory",directory=os.getcwd())
        print(os.listdir(r))
        print(r)
        ultimo=r.split('/')[-1]
        print('ult',ultimo)
        # recorremos cada file del nuevo directorio
        for file_name in os.listdir(r):
            source = r +'/'+ file_name
            destination = 'recetastextos/'+ultimo+'/' + file_name
            print('se va al destino',destination)
            #si existe el archivo de source lo movemos al destino

            if os.path.exists('recetastextos/'+ultimo)==False:
                os.makedirs('recetastextos/'+ultimo)
                shutil.move(source, destination)
                print('Moved:', file_name)
            else:
                #aqui va a haber un error
                shutil.move(source, destination)
                print('Moved:', file_name)
        #actualizamos el combobox
        self.cbcategoria.clear()
        self.cbcategoria.addItems(os.listdir('recetastextos/'))
    def aniadir_directorio(self):
        r=QFileDialog.getExistingDirectory(self, "Select Directory",directory=os.getcwd())
        self.varableSeleccionCarpetaGuardarModelo=r
    def volver(self):
        self.gui = Index()
        self.gui.show()
        self.gui.showMaximized()
        self.close()

    def guardarModelo(self, modeloEntrenado):
        if(self.varableSeleccionCarpetaGuardarModelo==""):
            print("no hay ruta")
        elif(self.formguardar.text()==""):
            print("no hay nombre de archivo")
        else:
            rutaGuardarModelo = self.varableSeleccionCarpetaGuardarModelo + "/" + self.formguardar.text() + ".pkl"
            joblib.dump(modeloEntrenado, rutaGuardarModelo)
            print(rutaGuardarModelo)





class Test(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eat Easer Test page")
        self.setWindowIcon(QIcon("imagenes/EatEaser-Logo.png"));
        # variables globales
        self.nombrecarpeta=''
        self.info=self.Informacion()
        self.varableRutaModeloEntrenado=""

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
        #self.path_btn.clicked.connect()
        #self.btn_guardar.clicked.connect(self.recuperarModeloEntrenado)
        self.nuevo.clicked.connect(self.aniadir_categoria)
        # form del grid guardar
        self.formguardar = QLineEdit()
        self.lform = QLabel("Guardar modelo:")
        self.lform.setStyleSheet(sform)

        # aniadimos los widgets a guardar
        self.guardar.addWidget(self.lform, 0, 0, 1, 1)
        self.guardar.addWidget(self.formguardar, 0, 1, 1, 1)
        self.guardar.addWidget(self.btn_guardar, 0, 2)
        self.guardar.addWidget(self.path_btn, 0, 3)

        # grid del grafico

        self.grafico = QVBoxLayout()


        #agregamos la tabla
        self.tableWidget = QTableWidget()
        self.btn_seleccion_modelo.clicked.connect(
            lambda: self.informacion('Modelo Seleccionado', 'Estos son sus archivos:'))
        self.btnalgoritmo.clicked.connect(self.vista_previa)




        self.grafico.addWidget(self.tableWidget)

        # boton de retorno izquierdo
        self.retorno = QPushButton(u"\u2190" + ' Main Page/ Test')
        self.retorno.setStyleSheet(sretorno)
        self.retorno.clicked.connect(self.volver)
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
        self.derlayout.addWidget(self.linfo, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(self.ltitulo, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(self.ldescrip, 2, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(self.vista, 3, 0, 3, 1, QtCore.Qt.AlignHCenter)
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



    class Informacion(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("EatEaser-Visualizar Texto")
            scategorias='font-family:"NSimSun";font-size:20px;border:1px solid black;border-radius:12px;'
            stexto='line-height: 0.9;font-family:"NSimSun";font-size:24px;background-color:white;border:1px solid black;text-align:justify;text-transform:capitalize;padding:20px;'
            sventana='background-color:black;color:white;font-family:"NSimSun";font-size:20px;text-align:center;'
            self.layout=QGridLayout()
            #self.varableRutaModeloEntrenado=""
            self.nombre=QLabel('Nombre')
            self.categoria=QLabel('Categoria')
            self.texto=QLabel('Texto')
            self.informacion= QScrollArea()
            self.texto.setWordWrap(True)
            self.texto.setStyleSheet(stexto)
            self.categoria.setStyleSheet(scategorias)
            self.nombre.setStyleSheet(scategorias)
            self.id_ventana=QLabel('Visualizacion de texto')
            self.id_ventana.setStyleSheet(sventana)
            self.informacion.setWidget(self.texto)
            self.informacion.setWidgetResizable(True)
            self.layout.addWidget(self.id_ventana, 0, 0, 1, 4)
            self.layout.addWidget(self.nombre, 1, 0,1,1,QtCore.Qt.AlignCenter)
            self.layout.addWidget(self.categoria, 1, 1,1,1,QtCore.Qt.AlignCenter)
            self.layout.addWidget(self.informacion, 2, 0,6,4)

            self.setLayout(self.layout)
            self.ruta=[]
            self.carpeta_seleccionada=''

        def ver_(self,list):
            with open(self.carpeta_seleccionada+'/'+self.ruta[list], "r") as archivo:
                for linea in archivo:
                    resultado=linea

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
        r = QFileDialog.getOpenFileName(parent=None, caption='Select Directory', directory=os.getcwd(), filter='Pickle files (*.pkl)')
        self.varableRutaModeloEntrenado =r [0]
        #print(self.varableRutaModeloEntrenado)
        
    def setData(self):
        if(self.varableRutaModeloEntrenado!=""):
            print("modelo cargado")
            modelo_entrenado = joblib.load(self.varableRutaModeloEntrenado)
        self.tableWidget.setRowCount(len(os.listdir(self.cbcategoria.placeholderText())))
        self.tableWidget.setColumnCount(3)
        self.info.ruta=os.listdir(self.cbcategoria.placeholderText())
        self.info.carpeta_seleccionada=self.cbcategoria.placeholderText()
        self.tableWidget.setHorizontalHeaderLabels(["Texto", "Categoria", "Ver Texto"])
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        i=0


        self.nombrecarpeta=self.cbcategoria.placeholderText().split('/')[-1]
        for key in os.listdir(self.cbcategoria.placeholderText()):
            boton=QPushButton()
            self.ver.addButton(boton)
            self.ver.setId(boton,i)
            boton.setIcon(QIcon('imagenes/ojo.png'))
            self.tableWidget.setItem(i,0,QTableWidgetItem(key))
            self.tableWidget.setItem(i, 1, QTableWidgetItem('key'))
            self.tableWidget.setCellWidget(i, 2, boton)
            i=i+1


        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()

        self.tableWidget.show()

    def aniadir_categoria(self):
        r = QFileDialog.getExistingDirectory(self, "Select Directory", directory=os.getcwd())
        self.cbcategoria.setPlaceholderText(r)


    def volver(self):
        self.gui = Index()
        self.gui.show()
        self.gui.showMaximized()
        self.close()


    def mensaje_error(self,mensaje):
        QMessageBox.critical(
            self,
            "Error",
            mensaje,
            buttons=QMessageBox.Discard | QMessageBox.NoToAll | QMessageBox.Ignore,
            defaultButton=QMessageBox.Discard,
        )
    def vista_previa(self):
        self.vista.setText('')
        i = 0
        self.total_archivos=0

        carpetas=''
        #verificamos si hay seleccionados

        if self.cbcategoria.placeholderText()=='' or self.ltitulo.text()=='Nombre Algoritmo' :

            self.mensaje_error('Campos vacios.')
        else:
            self.setData()
            size = len(os.listdir(self.cbcategoria.placeholderText()))

            self.total_archivos = size
            texto=self.nombrecarpeta + ': ' + str(size) + ' archivos\n'





            # le añado todos los que esten en listbox
            self.vista.setText(texto+'\n'+'TOTAL: ' + ': ' + str(self.total_archivos) + ' archivos\n')



class WebScraping:
    def __init__(self,kw):
        self.keyword=kw
        self.listaNombres=[]
        self.listaTiempos= []
        self.listaImagenes=[]
        self.listaPrecios = []
        self.listaURL=[]
    def conexionPaginaWebLidl(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
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
            driver.find_element(By.CLASS_NAME, "cookie-alert-extended-button").click()
            print("Boton aceptar coquies seleccionado")
        except:
            print("No sale boton de aceptar coquies")

    def sacarInfoLidl(self, driver):
        # Para entrar en el alimento que nosotros queremos
        escribirIngrediente = driver.find_element(By.CLASS_NAME, "inputField.js_mIngredientSearchGroup-input")
        escribirIngrediente.send_keys(self.keyword)
        # hay que darle un poco de tiempo para que despues de escribir seleccione el enter sino no lo ejecuta bien
        time.sleep(2)
        escribirIngrediente.send_keys(Keys.RETURN)

        time.sleep(2)
        # PARA COGER EL NOMBRE DE LA RECETA DEL PRODUCTO
        todasRecetas = driver.find_element(By.CLASS_NAME, "oRecipeFeed-resultContainer.js_oRecipeFeed-resultContainer")
        time.sleep(1)
        nombre = todasRecetas.find_elements(By.CLASS_NAME, "mRecipeTeaser-title")
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
        src = todasRecetas.find_elements(By.CLASS_NAME, "picture-img.mRecipeTeaser-image.lazyloaded")
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


import mmap
class Download(QtWidgets.QMainWindow):
    def __init__(self):
        super(Download, self).__init__()
        # Cargamos el .ui file
        uic.loadUi('download.ui', self)
        self.grid = self.findChild(QGridLayout, 'grid_descargas')
        col=0
        fila=0
        for i in range(4):
            print(fila)
            if(col<4):

                frame = QFrame()
                self.grid.addWidget(frame, fila, col)
                frame_grid = QVBoxLayout()
                frame.setLayout(frame_grid)
                grid_titulo = QVBoxLayout()
                grid_boton = QVBoxLayout()
                frame_grid.addLayout(grid_titulo)
                titulo=QLabel('Titulo de video')
                titulo.setStyleSheet('font-family:"Bahnschrift Light";font-size:16px;')
                grid_titulo.addWidget(titulo)
                frame_grid.addLayout(grid_boton)
                ver=QPushButton('Ver texto')
                ver.setStyleSheet('background-color:black;color:white;font-family:"Californian FB";font-size:16px;border-radius:20px;')
                ver.setFixedSize(100,60)
                grid_boton.addWidget(ver)
                frame.setStyleSheet('background-color:white;border-radius:20px;')
                col=col+1
            else:
                col=0
                fila=fila+1


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        # Cargamos el .ui file
        uic.loadUi('app.ui', self)

        #ponemos unos estilos
        scategorias='border-radius:100px;}QPushButton:hover{border:4px solid black;}'

        #llamamos a los botones
        self.btnplatos= self.findChild(QtWidgets.QPushButton, 'btnplatos')
        self.btnplatos.setStyleSheet("QPushButton{border-image:url(imagenes/platos.jpg);"+scategorias)
        self.btnplatos.setFixedSize(200, 200)
        self.btnverdura= self.findChild(QtWidgets.QPushButton, 'btnverdura')
        self.btnverdura.setStyleSheet("QPushButton{border-image:url(imagenes/verdura.jpg);"+scategorias)
        self.btnverdura.setFixedSize(200, 200)
        self.btnarroz= self.findChild(QtWidgets.QPushButton, 'btnarroz')
        self.btnarroz.setStyleSheet("QPushButton{border-image:url(imagenes/arroz.jpg);"+scategorias)
        self.btnarroz.setFixedSize(200, 200)
        self.btnpasta = self.findChild(QtWidgets.QPushButton, 'btnpasta')
        self.btnpasta.setStyleSheet("QPushButton{border-image:url(imagenes/pasta.jpg);"+scategorias)
        self.btnpasta.setFixedSize(200, 200)
        self.btnmarisco = self.findChild(QtWidgets.QPushButton, 'btnmarisco')
        self.btnmarisco.setStyleSheet("QPushButton{border-image:url(imagenes/marisco.jpg);"+scategorias)
        self.btnmarisco.setFixedSize(200, 200)
        self.btnpescado = self.findChild(QtWidgets.QPushButton, 'btnpescado')
        self.btnpescado.setStyleSheet("QPushButton{border-image:url(imagenes/pescado.jpg);"+scategorias)
        self.btnpescado.setFixedSize(200, 200)
        self.btnbebidas = self.findChild(QPushButton, 'btnbebidas')
        self.btnbebidas.setStyleSheet("QPushButton{border-image:url(imagenes/bebida.jpg);"+scategorias)
        self.btnbebidas.setFixedSize(200, 200)
        self.btncarne = self.findChild(QPushButton, 'btncarne')
        self.btncarne.setStyleSheet("QPushButton{background-image:url(imagenes/carne.jpg);"+scategorias)
        self.btncarne.setFixedSize(200, 200)
        self.busqueda = self.findChild(QLineEdit, 'busqueda')
        self.btnbuscar = self.findChild(QPushButton, 'buscar')
        self.btnbuscar.setIcon(QIcon('imagenes/lupa.png'))
        self.busqueda.setStyleSheet("QPushButton{border-radius:10px;border:1px solid black;background-color:transparent;")
        #self.volver = self.findChild(QPushButton, 'back')
        #self.volver.setIcon(QIcon('imagenes/menu.png'))
        self.txt_frame=self.findChild(QGridLayout, 'gridLayout_8')

        self.grupo_botones=QButtonGroup()
        self.grid_productos = self.findChild(QGridLayout, 'grid_productos')

        #agrego primer frame


        self.buscar_recetas('cebolla')
        self.buscar_texto('Carpeta Arroz')
        #ponemos acciones a los botones
        self.btnbuscar.clicked.connect(lambda: self.buscar_recetas(self.busqueda.text()))
        self.btncarne.clicked.connect(lambda :self.buscar_recetas('carne'))
        self.btnpasta.clicked.connect(lambda: self.buscar_recetas('pasta'))
        self.btnpescado.clicked.connect(lambda: self.buscar_recetas('pescado'))
        self.btnmarisco.clicked.connect(lambda: self.buscar_recetas('marisco'))
        self.btnbebidas.clicked.connect(lambda: self.buscar_recetas('bebida'))
        self.btnplatos.clicked.connect(lambda: self.buscar_recetas('pan'))
        self.btnverdura.clicked.connect(lambda: self.buscar_recetas('lechuga'))
        #ponemos un default de recetas
        #self.buscar_recetas('cebolla')

        self.setStyleSheet('background-color:white;')
        self.show()
    def buscar_texto(self,categoria):
        directorio=os.listdir('recetastextos/'+categoria)

        j=0
        fila=0
        for i,texto in enumerate(directorio):
            #quiero que sean 10 columnas

            if j<10:

                self.boton = QPushButton(texto)
                self.boton.setStyleSheet('QPushButton{border-radius:20px;border:1px solid black;}QPushButton:hover{border:1px solid white;background-color:black;color:white;}')
                self.txt_frame.addWidget(self.boton,fila,j)
                j=j+1
            else:
                j=0
                fila=fila+1



    def buscar_productos(self,producto):
        ws = WebScraping(producto)
        cont = 0


        ws.conexionPaginaWebAhorraMas()
        for i, element in enumerate(ws.listaNombres):
            print('imagen')
            # si la columna ya va a mas de uno
            if (cont == 4):
                break
            else:
                imagen = self.findChild(QLabel, 'imgs_' + str(i))
                nombre = self.findChild(QLabel, 'nm_' + str(i))
                descripcion = self.findChild(QLabel, 'prc_' + str(i))

                nombre.setText(element)
                descripcion.setText(ws.listaPrecios[i])
                data = urllib.request.urlopen(ws.listaImagenes[i]).read()
                image = QtGui.QImage()
                image.loadFromData(data)
                pix = QtGui.QPixmap(image)
                imagen.setPixmap(pix)
                imagen.setScaledContents(True)

            cont = cont + 1

    def buscar_recetas(self,categoria):
        j = 0
        fila = 2
        ws2 = WebScraping(categoria)
        ws2.conexionPaginaWebLidl()
        for i, element in enumerate(ws2.listaNombres):
            if j < 4:
                img = True
                try:
                    imagen = QPushButton('')
                    response = requests.get(ws2.listaImagenes[i])
                    if response.status_code == 200:
                        with open("sample" + str(i) + ".jpg", 'wb') as f:
                            f.write(response.content)
                    imagen.setStyleSheet("border-image:url(sample" + str(i) + ".jpg);border-radius:100%;")
                    imagen.setFixedSize(200, 200)

                except:
                    print('imagen no obtenida')
                    img = False
                print(img)
                if (img == True):
                    n = QFrame()
                    n2 = QFrame()
                    n3 = QFrame()
                    n4 = QFrame()
                    vlt = QVBoxLayout()
                    n.setLayout(vlt)
                    self.grid_productos.addWidget(n, fila, j)
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
                    n3.layout().addWidget(QLabel(ws2.listaTiempos[i]))
                    n4.layout().addWidget(QLabel('♡'))
                    btn = QPushButton('')

                    btn.setIcon(QIcon('imagenes/exterior.png'))

                    btn.setIconSize(QSize(20, 20))
                    n4.layout().addWidget(btn)

                    # parte del medio
                    vlt.addWidget(n3)
                    vlt.addWidget(n4)
                    # parte de abajo

                    n.setStyleSheet(
                        'background-color:#ede8e1;border:1px solid black;font-family:"Segoe UI Semibold";font-size:16px;')
                    n2.setStyleSheet('background-color:white;text-decoration: underline;border:1px solid white;')
                    n3.setStyleSheet('background-color:white;text-decoration: underline;border:1px solid white;')
                    n4.setStyleSheet('background-color:white;text-decoration: underline;border:1px solid white;')
                    j = j + 1
            else:
                j = 0
                fila = fila + 1

        self.buscar_productos(categoria)




if __name__=='__main__':
    '''
    df=pd.DataFrame()
    df['receta']=None
    df['clasif']=None
    procesarDocs=ProcesarDocumentos()
    listaTextosCarpeta=procesarDocs.lectura()
    for index,content in enumerate(listaTextosCarpeta):
        for i in range(len(content)):
            text=procesarDocs.tratamientoTextos(listaTextosCarpeta[index][i])
            df=df.append({'receta':text,'clasif':index},ignore_index=True)
    '''
    app=QApplication(sys.argv)

    gui=Index()
    gui.show()
    gui.showMaximized()

    sys.exit(app.exec_())

