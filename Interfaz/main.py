import os
import sys
import shutil
import urllib.request

from PyQt5 import QtCore, QtWidgets, QtGui, Qt
from PyQt5.QtCore import QSize


import time
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase, QIcon, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QGridLayout, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, \
    QWidget, QSizePolicy, QComboBox, QLayout, QFormLayout, QLineEdit, QButtonGroup, QRadioButton, QCheckBox, \
    QFileDialog, QMessageBox, QTableWidget, QAbstractItemView, QTableWidgetItem, QHeaderView, QScrollArea


class Index(QWidget):
    def __init__(self):
        super().__init__()
        self.state='main'
        # Layout grande
        self.layout = QHBoxLayout()

        # lado izquierdo del grid
        self.izqlayout = QGridLayout()
        self.setWindowTitle("Eat Easer Main page")
        self.label = QLabel()
        self.pixmap = QPixmap('imagenes/imagen.jpg')
        self.label.setPixmap(self.pixmap)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.izqlayout.addWidget(self.label,0,0)
        # lado derecho del grid
        self.derlayout = QGridLayout()
        self.der_grid = QGridLayout()

        #Stylesheets
        botones_menu = "QPushButton{ color :black;background-color:white;border:3px solid white;font-weight:lighter;font-size:22px;font-family:'Bahnschrift Light';letter-spacing:3px;}QPushButton:hover {color:gray}"
        stitulo="margin:0,0,0,0;white-space: normal;color:black;font-size:34px;font-weight:bold;text-transform:upperrcase;text-align: center;"
        sdescripcion="color:gray;font-size:24px;font-weight:light;text-transform:lowercase;text-align: center;"
        sacceder="QPushButton{border:1px solid white;background-color:white;font-size:18px;font-family:'Bahnschrift Light';}QPushButton:hover{font-weight:bold}"
        sfuente_1="font-family:'Bahnschrift Light';letter-spacing:3px;"
        snombres = "font-size:24px;font-family:'Bahnschrift Light';letter-spacing:3px;"
        #botones menu
        self.about = QtWidgets.QPushButton("About", self)
        self.app = QtWidgets.QPushButton("Application", self)
        self.train = QtWidgets.QPushButton("Train", self)
        self.test = QtWidgets.QPushButton("Test", self)
        #botones menu estilizados
        self.about.setStyleSheet(botones_menu)
        self.train.setStyleSheet(botones_menu)
        self.test.setStyleSheet(botones_menu)
        self.app.setStyleSheet(botones_menu)
        #grid de arriba añadimos menu
        self.arriba_grid = QGridLayout()
        self.arriba_grid.addWidget(self.about, 0, 0)
        self.arriba_grid.addWidget(self.train, 0, 1)
        self.arriba_grid.addWidget(self.test, 0, 2)
        self.arriba_grid.addWidget(self.app, 0, 3)
        #grid del medio
        self.medio_grid = QGridLayout()
        self.abajo_grid = QGridLayout()
        #titulo del main
        self.titulo = QLabel(self)
        self.titulo.setText("Bienvenido al Startup de Eateaser")
        self.titulo.setStyleSheet(sfuente_1+stitulo)
        #descripcion del main
        self.descripcion = QLabel(self)
        self.descripcion.setText("Selecciona una opción del menú y descubre sobre ello.")
        self.descripcion.setStyleSheet(sfuente_1+sdescripcion)
        #Boton oculto de acceder
        self.acceder=QPushButton('Acceder')
        self.acceder.setVisible(False)
        self.acceder.setIcon(QIcon('imagenes/up-arrow.png'))
        self.acceder.setStyleSheet(sacceder)
        #agrid del medio aniadimos titulo, descripcion y boton
        self.medio_grid.addWidget(self.titulo,0,0,1,1,QtCore.Qt.AlignHCenter)
        self.medio_grid.addWidget(self.descripcion,1,0,1,1,QtCore.Qt.AlignHCenter)
        self.medio_grid.addWidget(self.acceder, 2, 0,2,1, QtCore.Qt.AlignHCenter)
        #grid derecho aniadimos todos los grids
        self.der_grid = QGridLayout()
        self.der_grid.addLayout(self.arriba_grid, 0, 0)
        self.der_grid.addLayout(self.medio_grid, 1, 0)
        self.der_grid.addLayout(self.abajo_grid,2,0)
        self.derlayout.addLayout(self.der_grid, 2, 0,3,0)
        # añadimos todo
        self.layout.addLayout(self.izqlayout,20)
        self.layout.addLayout(self.derlayout,20)
        self.setLayout(self.layout)
        #configuraciones de la pagina
        self.setStyleSheet("background-color :  white")
        self.showMaximized()
        #acciones de botones
        self.about.clicked.connect(lambda :self.menuClicked('Startup Eateaser','Compañia encargada para sugerirte las mejores recetas.\nSeremos tus aliados a la hora de cocinar.\nNosotros te permitimos una aplicacion facil\npara conocer la clasificacion de \n'
                'tus platillos favoritos. Ademas tambien \nclasificamos resetas y te enseñamos nuestros algoritmos',False,True))
        self.train.clicked.connect(lambda :self.menuClicked('Fase de Entrenamiento','Compañia encargada para sugerirte las mejores recetas.\nSeremos tus aliados a la hora de cocinar.\nNosotros te permitimos una aplicacion facil\npara conocer la clasificacion de \n'
                'tus platillos favoritos. Ademas tambien \nclasificamos resetas y te enseñamos nuestros algoritmos',True,False))
        self.test.clicked.connect(lambda :self.menuClicked('Fase de Testeo','Compañia encargada para sugerirte las mejores recetas.\nSeremos tus aliados a la hora de cocinar.\nNosotros te permitimos una aplicacion facil\npara conocer la clasificacion de \n'
                'tus platillos favoritos. Ademas tambien \nclasificamos resetas y te enseñamos nuestros algoritmos',True,False))
        self.app.clicked.connect(lambda :self.menuClicked('Aplicacion','Compañia encargada para sugerirte las mejores recetas.\nSeremos tus aliados a la hora de cocinar.\nNosotros te permitimos una aplicacion facil\npara conocer la clasificacion de \n'
                'tus platillos favoritos. Ademas tambien \nclasificamos resetas y te enseñamos nuestros algoritmos',True,False))
        self.acceder.clicked.connect(self.openTrain)

        #about informacion botones imagenes
        self.juancar = QPushButton()
        self.adi = QPushButton()
        self.carlos = QPushButton()
        self.rober = QPushButton()
        #personalizamos los botones acontinuacion
        self.personalizar_boton( self.juancar, 'juancar.jpeg')
        self.personalizar_boton(self.adi, 'adi.jpeg')
        self.personalizar_boton(self.carlos, 'carlos.jpeg')
        self.personalizar_boton(self.rober, 'rober.jpeg')
        #labels personalizadas
        self.ljuancar = QLabel('Juan\nCarlos')
        self.ladi = QLabel('Adilem\nDobras')
        self.lcarlos = QLabel('Carlos\nGonzales')
        self.lrober = QLabel('Roberto\nEchevarria')
        #estilizamos las labels
        self.ladi.setStyleSheet(snombres)
        self.ljuancar.setStyleSheet(snombres)
        self.lrober.setStyleSheet(snombres)
        self.lcarlos.setStyleSheet(snombres)
        #grid de abajo aniadimos widgets
        self.abajo_grid.addWidget(self.juancar, 0, 0)
        self.abajo_grid.addWidget(self.adi, 0, 1)
        self.abajo_grid.addWidget(self.carlos, 0, 2)
        self.abajo_grid.addWidget(self.rober, 0, 3)
        self.abajo_grid.addWidget(self.ljuancar, 1, 0)
        self.abajo_grid.addWidget(self.ladi, 1, 1)
        self.abajo_grid.addWidget(self.lcarlos, 1, 2)
        self.abajo_grid.addWidget(self.lrober, 1, 3,1,1)
        self.abajo_grid.setContentsMargins(100,0,100,0)

        self.apagar_widgets(False)
    def apagar_widgets(self,boolean):
        self.carlos.setVisible(boolean)
        self.juancar.setVisible(boolean)
        self.adi.setVisible(boolean)
        self.rober.setVisible(boolean)
        self.ladi.setVisible(boolean)
        self.lrober.setVisible(boolean)
        self.ljuancar.setVisible(boolean)
        self.lcarlos.setVisible(boolean)


    def personalizar_boton(self,boton,nombre):
        boton.setIcon(QIcon('imagenes/'+nombre))
        boton.setIconSize(QSize(200, 200))
        boton.setFixedSize(200, 200)
        boton.setStyleSheet('border-radius:12px;')
    def menuClicked(self,titulo,descripcion,acceder,widgets):
        self.titulo.setText(titulo)
        self.descripcion.setText(descripcion)
        self.acceder.setVisible(acceder)
        self.apagar_widgets(widgets)
        self.state = titulo
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
class Train(QWidget):
    def __init__(self):
        super().__init__()
        #variables globales
        self.seleccionados = []
        self.checkboxes = []

        #layout grande
        self.layout = QGridLayout()
        #partes del layout grande
        self.izqlayout=QGridLayout()
        self.derlayout = QGridLayout()
        #stylesheet
        scategoria="font-family:'Bahnschrift Light';font-size:24px;letter-spacing:3px;padding:0%;padding:5px;"
        scbcategoria="color :black;background-color:white;border-bottom:3px solid black;font-weight:lighter;font-size:22px;font-family:'Bahnschrift Light';letter-spacing:3px;"
        sbtnruta='QPushButton{background-color:transparent;border:1px solid transparent}QPushButton:hover{border:1px solid black;border-radius:12px;}'
        sbotones = 'QPushButton{border:transparent;background-color:transparent;}QPushButton:hover{border:2px solid black;border-radius:12px;}'
        salgoritmo="font-family:'Bahnschrift Light';font-size:24px;letter-spacing:3px;padding:0%"
        sbtnalgoritmo = 'QPushButton{color:white;border-radius:12px;background-color:black;margin:0;font-family:"Bahnschrift Light";font-size:24px;}QPushButton:hover{color:black;background-color:transparent;border:2px solid black;}'
        sform="font-family:'Bahnschrift Light';font-style:italic;font-size:24px;letter-spacing:3px;padding:0%"
        sinfo='background-color:white;border-radius:12px;border:1px white;'
        stextos_derecha = 'font-family:"NSimSun";font-size:24px;overflow:hidden;white-space: nowrap;'
        sretorno='font-family:"NSimSun";font-size:24px;overflow:hidden;white-space: nowrap;color:white;background-color:black;'
        #grid de la ruta
        self.rutalayout= QGridLayout()
        #labels de ruta
        self.lcategoria=QLabel('Selecciona la categoria')
        self.lcategoria.setStyleSheet(scategoria)
        #combobox de ruta
        self.cbcategoria=QComboBox()
        self.cbcategoria.setFixedSize(800,40)
        self.cbcategoria.addItems(os.listdir('recetastextos/'))
        self.cbcategoria.setStyleSheet(scbcategoria)
        #botones de ruta
        self.anadir = QPushButton()
        self.anadir.setIcon(QIcon('imagenes/cargar.png'))
        self.anadir.setFixedSize(QtCore.QSize(40, 40))
        self.nuevo = QPushButton()
        self.nuevo.setIcon(QIcon('imagenes/add.png'))
        self.nuevo.setFixedSize(QtCore.QSize(40, 40))
        # estilizamos los botones
        self.nuevo.setStyleSheet(sbtnruta)
        self.anadir.setStyleSheet(sbtnruta)
        #aniadimos al layout de ruta
        self.rutalayout.addWidget(self.lcategoria, 0, 0, 1, 1)
        self.rutalayout.addWidget(self.cbcategoria,0,1,1,1)
        self.rutalayout.addWidget(self.anadir, 0, 2, 1, 1)
        self.rutalayout.addWidget(self.nuevo,0,3,1,1)



        #grid de algoritmos
        self.algoritmolayout = QGridLayout()
        #labels algoritmo
        self.lalgoritmo = QLabel('Algoritmo:')
        self.lalgoritmo.setStyleSheet(salgoritmo)
        #botones de algorimos
        self.btn_knn=QPushButton('K-NN')
        self.btn_rf = QPushButton('Random-Forest')
        self.btn_rn = QPushButton('Red Neuronal')
        self.btnalgoritmo = QPushButton()
        #estilizamos botones
        self.btn_knn.setFixedSize(QtCore.QSize(400, 80))
        self.btn_rf.setFixedSize(QtCore.QSize(400, 80))
        self.btn_rn.setFixedSize(QtCore.QSize(400, 80))
        self.btn_knn.setStyleSheet(sbtnalgoritmo)
        self.btn_rf.setStyleSheet(sbtnalgoritmo)
        self.btn_rn.setStyleSheet(sbtnalgoritmo)
        size = QSize(50, 50)
        self.btnalgoritmo.setIconSize(size)
        self.btnalgoritmo.setStyleSheet(sbotones)
        self.btnalgoritmo.setIcon(QIcon('imagenes/boton-de-play.png'))
        self.btnalgoritmo.setFixedSize(QtCore.QSize(80, 80))

        #aniadimos al grid de algoritmos
        self.algoritmolayout.addWidget(self.lalgoritmo,0,0,1,4)
        self.algoritmolayout.addWidget(self.btn_knn,1,0,2,1)
        self.algoritmolayout.addWidget(self.btn_rf,1,1,2,1)
        self.algoritmolayout.addWidget(self.btn_rn,1,2,2,1)
        self.algoritmolayout.addWidget(self.btnalgoritmo,1,3,2,1)

        #grjd guardar
        self.guardar=QGridLayout()
        self.seleccionlayout=QHBoxLayout()

        #botones de grid guardar
        self.borrar =QPushButton()
        self.path_btn = QPushButton('')
        self.btn_guardar = QPushButton()

        #estilizamos botones
        self.borrar.setFixedSize(QtCore.QSize(80, 80))
        self.borrar.setIcon(QIcon('imagenes/delete.png'))
        self.path_btn.setIcon(QIcon('imagenes/lupa.png'))
        self.btn_guardar.setIcon(QIcon('imagenes/guardar-el-archivo.png'))
        self.btn_guardar.setStyleSheet(sbotones)
        self.path_btn.setStyleSheet(sbotones)

        #aniadimos a la seleccion
        self.seleccionlayout.addWidget(self.borrar,1,QtCore.Qt.AlignLeft)

        # eventos de botones
        self.anadir.clicked.connect(self.aniadir_boton)
        self.borrar.clicked.connect(self.eliminar_boton)
        self.path_btn.clicked.connect(self.aniadir_directorio)
        self.btn_knn.clicked.connect(
            lambda: self.informacion('Algoritmo K-NN', 'Este algoritmo hace esto y esto y esto'))
        self.btn_rn.clicked.connect(
            lambda: self.informacion('Algoritmo Red Neuronal', 'Este algoritmo hace esto y esto y esto'))
        self.btn_rf.clicked.connect(
            lambda: self.informacion('Algoritmo Random Forest', 'Este algoritmo hace esto y esto y esto'))
        self.btnalgoritmo.clicked.connect(self.vista_previa)
        self.nuevo.clicked.connect(self.aniadir_categoria)
        #form del grid guardar
        self.formguardar = QLineEdit()
        self.lform=QLabel("Guardar modelo:")
        self.lform.setStyleSheet(sform)

        #aniadimos los widgets a guardar
        self.guardar.addWidget(self.lform, 0, 0, 1, 1)
        self.guardar.addWidget(self.formguardar, 0, 1, 1, 1)
        self.guardar.addWidget(self.btn_guardar, 0, 2)
        self.guardar.addWidget(self.path_btn, 0, 3)


        # grid del grafico
        self.layout13 = QLabel('d')
        self.layout13.setStyleSheet('background-color:white')
        self.grafico = QVBoxLayout()
        self.grafico.addWidget(self.layout13)

        #boton de retorno izquierdo
        self.retorno=QPushButton(u"\u2190"+' Main Page/ Entrenamiento')
        self.retorno.setStyleSheet(sretorno)
        self.retorno.clicked.connect(self.volver)
        #aniadimos los layouts al lado izq
        self.izqlayout.addWidget(self.retorno, 0, 0)
        self.izqlayout.addLayout(self.rutalayout,1,0)
        self.izqlayout.addLayout(self.algoritmolayout,2,0)
        self.izqlayout.addLayout(self.seleccionlayout,3, 0)
        self.izqlayout.addLayout(self.grafico,4, 0)
        self.izqlayout.addLayout(self.guardar,5,0)
        #estilizamos los layouts
        self.izqlayout.setRowStretch(0, 1)
        self.izqlayout.setRowStretch(1, 1)
        self.izqlayout.setRowStretch(2, 1)
        self.izqlayout.setRowStretch(3, 1)
        self.izqlayout.setRowStretch(4, 2)
        self.izqlayout.setRowStretch(5, 1)

        #zona derecha del layout labels
        self.linfo=QPushButton()
        self.ltitulo=QLabel('Nombre Algoritmo')
        self.ldescrip = QLabel('Descripcion Algoritmo')
        self.vista = QLabel('Vista Algoritmo')
        self.fondo = QLabel()
        #estilizar labels
        self.linfo.setIcon(QIcon('imagenes/informacion.png'))
        self.linfo.setStyleSheet(sinfo)
        self.linfo.setFixedSize(QtCore.QSize(400, 80))
        size = QSize(50, 50)
        self.linfo.setIconSize(size)
        self.fondo.setStyleSheet(sinfo)

        #aniadimos widgets en lado derecho
        self.derlayout.addWidget(self.fondo,0,0,6,1)
        self.derlayout.addWidget(self.linfo,0,0,1,1,QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(self.ltitulo, 1, 0, 1, 1,QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(self.ldescrip, 2, 0, 1, 1,QtCore.Qt.AlignHCenter)
        self.derlayout.addWidget(self.vista, 3, 0, 3, 1,QtCore.Qt.AlignHCenter)
        self.derlayout.rowStretch(1)

        #aniadimos los layouts al total
        self.layout.addLayout(self.izqlayout,0,0)
        self.layout.addLayout(self.derlayout,0,1)
        self.layout.setColumnStretch(0,3)
        self.layout.setColumnStretch(1, 1)

        #estilizamos zona derecha
        self.ltitulo.setStyleSheet(stextos_derecha)
        self.ldescrip.setStyleSheet(stextos_derecha)
        self.vista.setStyleSheet(stextos_derecha)

        self.setLayout(self.layout)

    def informacion(self,titulo,descripcion):
            self.ltitulo.setText(titulo)
            self.ldescrip.setText(descripcion)

    def vista_previa(self):
        self.vista.setText('')
        i = 0
        self.total_archivos=0

        carpetas=''
        #verificamos si hay seleccionados

        if len(self.seleccionados)==0 or self.ltitulo.text()=='Nombre Algoritmo' :

            self.mensaje_error('Campos vacios.')
        else:
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
        print(r)
        self.formguardar.setPlaceholderText(r)
    def volver(self):
        self.gui = Index()
        self.gui.show()
        self.gui.showMaximized()
        self.close()




class Test(QWidget):
    def __init__(self):
        super().__init__()
        # variables globales
        self.nombrecarpeta=''
        self.info=self.Informacion()

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
        self.btn_knn = QPushButton('K-NN')
        self.btn_rf = QPushButton('Random-Forest')
        self.btn_rn = QPushButton('Red Neuronal')
        self.btnalgoritmo = QPushButton()
        # estilizamos botones
        self.btn_knn.setFixedSize(QtCore.QSize(400, 80))
        self.btn_rf.setFixedSize(QtCore.QSize(400, 80))
        self.btn_rn.setFixedSize(QtCore.QSize(400, 80))
        self.btn_knn.setStyleSheet(sbtnalgoritmo)
        self.btn_rf.setStyleSheet(sbtnalgoritmo)
        self.btn_rn.setStyleSheet(sbtnalgoritmo)
        size = QSize(50, 50)
        self.btnalgoritmo.setIconSize(size)
        self.btnalgoritmo.setStyleSheet(sbotones)
        self.btnalgoritmo.setIcon(QIcon('imagenes/boton-de-play.png'))
        self.btnalgoritmo.setFixedSize(QtCore.QSize(80, 80))

        # aniadimos al grid de algoritmos
        self.algoritmolayout.addWidget(self.lalgoritmo, 0, 0, 1, 4)
        self.algoritmolayout.addWidget(self.btn_knn, 1, 0, 2, 1)
        self.algoritmolayout.addWidget(self.btn_rf, 1, 1, 2, 1)
        self.algoritmolayout.addWidget(self.btn_rn, 1, 2, 2, 1)
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



        self.btn_knn.clicked.connect(
            lambda: self.informacion('Algoritmo K-NN', 'Este algoritmo hace esto y esto y esto'))
        self.btn_rn.clicked.connect(
            lambda: self.informacion('Algoritmo Red Neuronal', 'Este algoritmo hace esto y esto y esto'))
        self.btn_rf.clicked.connect(
            lambda: self.informacion('Algoritmo Random Forest', 'Este algoritmo hace esto y esto y esto'))
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
    def setData(self):

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
class Aplicacion(QMainWindow):
    def __init__(self):
        super(Aplicacion, self).__init__()

        self.scroll = QScrollArea()  # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()  # Widget that contains the collection of Vertical Box
        self.layout = QVBoxLayout()  # The Vertical Box that contains the Horizontal Boxes of  labels and buttons

        self.lytcategorias = QGridLayout()

        self.btncarnes = QPushButton()
        self.btnplatos = QPushButton()
        self.btnbebidas = QPushButton()
        self.btnmarisco = QPushButton()
        self.btnpasta = QPushButton()
        self.btnverdura = QPushButton()
        self.btnarroz = QPushButton()
        self.btnpescado = QPushButton()
        # self.btncarnes.setIcon(QIcon('imagenes/carne.jpg'))

        # self.btncarnes.setIconSize(QSize(200, 200))
        self.btnplatos.setStyleSheet(
            "QPushButton{border-image:url(imagenes/platos.jpg);border-radius:100px;}QPushButton:hover{border:4px solid black;}")
        self.btnplatos.setFixedSize(200, 200)
        self.btnverdura.setStyleSheet("border-image:url(imagenes/verdura.jpg);border-radius:100px")
        self.btnverdura.setFixedSize(200, 200)
        self.btnarroz.setStyleSheet("border-image:url(imagenes/arroz.jpg);border-radius:100px")
        self.btnarroz.setFixedSize(200, 200)
        self.btnpasta.setStyleSheet("border-image:url(imagenes/pasta.jpg);border-radius:100px")
        self.btnpasta.setFixedSize(200, 200)
        self.btnmarisco.setStyleSheet("border-image:url(imagenes/marisco.jpg);border-radius:100px")
        self.btnmarisco.setFixedSize(200, 200)
        self.btnpescado.setStyleSheet("border-image:url(imagenes/pescado.jpg);border-radius:100px")
        self.btnpescado.setFixedSize(200, 200)
        self.btnbebidas.setStyleSheet("border-image:url(imagenes/bebida.jpg);border-radius:100px")
        self.btnbebidas.setFixedSize(200, 200)
        self.btncarnes.setStyleSheet("border-image:url(imagenes/carne.jpg);border-radius:100px")
        self.btncarnes.setFixedSize(200, 200)
        self.lytcategorias.addWidget(self.btncarnes, 0, 0, 1, 1)
        self.lytcategorias.addWidget(self.btnpescado, 0, 1)
        self.lytcategorias.addWidget(self.btnverdura, 0, 2)
        self.lytcategorias.addWidget(self.btnmarisco, 0, 3)
        self.lytcategorias.addWidget(self.btnpasta, 0, 4)
        self.lytcategorias.addWidget(self.btnbebidas, 0, 5)
        self.lytcategorias.addWidget(self.btnplatos, 0, 6)
        self.lytcategorias.addWidget(self.btnarroz, 0, 7)
        self.lytcategorias.addWidget(QLabel('Carnes'), 1, 0, 1, 1, QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Pescados'), 1, 1, QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Verduras'), 1, 2, QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Marisco'), 1, 3, QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Pasta'), 1, 4, QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Bebidas'), 1, 5, QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Platos Menores'), 1, 6, QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Arroz'), 1, 7, QtCore.Qt.AlignCenter)
        self.layout.addLayout(self.lytcategorias)

        self.discover = QGridLayout()
        self.ldiscover = QLabel('Discover new favorites')
        self.discover.addWidget(self.ldiscover)
        self.discover_items = QGridLayout()

        self.layout.addLayout(self.discover)
        ws = WebScraping()
        driver = ws.conexionPaginaWebAhorraMas("pimiento")
        cont = 0

        for i, element in enumerate(ws.listaNombres):
            # si la columna ya va a mas de uno
            if (cont == 4):
                break
            else:
                nombre = QLabel(element)
                nombre.setStyleSheet('font-family:"NSimSun";font-size:20px;background-color:white;')
                precio = QLabel(ws.listaPrecios[i])
                precio.setStyleSheet('font-family:"NSimSun";font-size:20px;background-color:white;')
                # imagen=QPushButton()

                data = urllib.request.urlopen(ws.listaImagenes[i]).read()

                # imagen.setStyleSheet('border-image:url('+ws.listaImagenes[i]+');')
                image = QtGui.QImage()
                image.loadFromData(data)

                lbl = QLabel()
                pix = QtGui.QPixmap(image)

                lbl.setPixmap(pix.scaled(350, 200))
                lbl.setScaledContents(True)
                # imagen.setFixedSize(100,100)
                self.discover_items.addWidget(lbl, 0, i)
                self.discover_items.addWidget(nombre, 1, i)
                self.discover_items.addWidget(precio, 2, i)
            cont = cont + 1
        self.layout.addLayout(self.discover_items)
        self.featured = QGridLayout()
        # self.featured.addWidget(self.informacion,0,1,4,1)
        self.featured_items = QGridLayout()

        # self.informacion.setLayout(self.featured_items)

        # self.informacion.setWidgetResizable(True)
        # self.informacion.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)"""
        for i in range(1, 50):
            object = QLabel("TextLabel: " + str(i))
            self.vbox.addWidget(object)


        self.widget.setLayout(self.layout)

        # Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

        self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle('Scroll Area Demonstration')
        self.show()
        """
        scategorias='border:1px solid black;border-radius:100px;background-color:white;background-image:url("imagenes/carne.jpg");'


        self.lytcategorias=QGridLayout()


        self.btncarnes=QPushButton() 
        self.btnplatos=QPushButton()
        self.btnbebidas=QPushButton()
        self.btnmarisco=QPushButton()
        self.btnpasta=QPushButton()
        self.btnverdura=QPushButton()
        self.btnarroz=QPushButton()
        self.btnpescado=QPushButton()
        #self.btncarnes.setIcon(QIcon('imagenes/carne.jpg'))

        #self.btncarnes.setIconSize(QSize(200, 200))
        self.btnplatos.setStyleSheet("QPushButton{border-image:url(imagenes/platos.jpg);border-radius:100px;}QPushButton:hover{border:4px solid black;}")
        self.btnplatos.setFixedSize(200, 200)
        self.btnverdura.setStyleSheet("border-image:url(imagenes/verdura.jpg);border-radius:100px")
        self.btnverdura.setFixedSize(200, 200)
        self.btnarroz.setStyleSheet("border-image:url(imagenes/arroz.jpg);border-radius:100px")
        self.btnarroz.setFixedSize(200, 200)
        self.btnpasta.setStyleSheet("border-image:url(imagenes/pasta.jpg);border-radius:100px")
        self.btnpasta.setFixedSize(200, 200)
        self.btnmarisco.setStyleSheet("border-image:url(imagenes/marisco.jpg);border-radius:100px")
        self.btnmarisco.setFixedSize(200, 200)
        self.btnpescado.setStyleSheet("border-image:url(imagenes/pescado.jpg);border-radius:100px")
        self.btnpescado.setFixedSize(200, 200)
        self.btnbebidas.setStyleSheet("border-image:url(imagenes/bebida.jpg);border-radius:100px")
        self.btnbebidas.setFixedSize(200, 200)
        self.btncarnes.setStyleSheet("border-image:url(imagenes/carne.jpg);border-radius:100px")
        self.btncarnes.setFixedSize(200,200)
        self.lytcategorias.addWidget(self.btncarnes,0,0,1,1)
        self.lytcategorias.addWidget(self.btnpescado,0,1)
        self.lytcategorias.addWidget(self.btnverdura,0,2)
        self.lytcategorias.addWidget(self.btnmarisco,0,3)
        self.lytcategorias.addWidget(self.btnpasta,0,4)
        self.lytcategorias.addWidget(self.btnbebidas,0,5)
        self.lytcategorias.addWidget(self.btnplatos,0,6)
        self.lytcategorias.addWidget(self.btnarroz,0,7)
        self.lytcategorias.addWidget(QLabel('Carnes'), 1, 0, 1, 1,QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Pescados'), 1, 1,QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Verduras'), 1, 2,QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Marisco'), 1, 3,QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Pasta'), 1, 4,QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Bebidas'), 1, 5,QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Platos Menores'), 1, 6,QtCore.Qt.AlignCenter)
        self.lytcategorias.addWidget(QLabel('Arroz'), 1, 7,QtCore.Qt.AlignCenter)
        self.layout.addLayout(self.lytcategorias,0,0,1,2)
        self.informacion=QScrollArea()
        self.discover=QGridLayout()
        self.ldiscover=QLabel('Discover new favorites')
        self.discover.addWidget(self.ldiscover)
        self.discover_items=QGridLayout()

        self.layout.addLayout(self.discover, 3, 0, 1, 1)


        ws = WebScraping()
        driver = ws.conexionPaginaWebAhorraMas("pimiento")
        cont=0

        for i,element in enumerate(ws.listaNombres):
            #si la columna ya va a mas de uno
            if(cont==4):
                break
            else:
                nombre=QLabel(element)
                nombre.setStyleSheet('font-family:"NSimSun";font-size:20px;background-color:white;')
                precio = QLabel(ws.listaPrecios[i])
                precio.setStyleSheet('font-family:"NSimSun";font-size:20px;background-color:white;')
                #imagen=QPushButton()

                data = urllib.request.urlopen(ws.listaImagenes[i]).read()

                #imagen.setStyleSheet('border-image:url('+ws.listaImagenes[i]+');')
                image = QtGui.QImage()
                image.loadFromData(data)

                lbl =QLabel()
                pix=QtGui.QPixmap(image)

                lbl.setPixmap(pix.scaled(350,200))
                lbl.setScaledContents(True)
                #imagen.setFixedSize(100,100)
                self.discover_items.addWidget(lbl,0,i)
                self.discover_items.addWidget(nombre,1,i)
                self.discover_items.addWidget(precio, 2, i)
            cont=cont+1
        self.layout.addLayout(self.discover_items, 3, 1, 1, 1)
        self.featured=QGridLayout()
        #self.featured.addWidget(self.informacion,0,1,4,1)
        self.featured_items=QGridLayout()

        #self.informacion.setLayout(self.featured_items)

        #self.informacion.setWidgetResizable(True)
        #self.informacion.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)"""
        """for i,element in enumerate(ws.listaNombres):
            self.featured_items.addWidget(QLabel(element),i,0,3,2)
            self.featured_items.addWidget(QLabel(element), i, 1,3,2)"""

        """#self.informacion.setWidget(self)

        #self.layout.addWidget(self.informacion)
        self.layout.addLayout(self.featured_items, 4, 0, 1, 5)"""




import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

import time


class WebScraping:
    def __init__(self):
        self.listaNombres=[]
        self.listaTiempos= []
        self.listaImagenes=[]
        self.listaPrecios = []
        self.listaURL=[]
    def conexionPaginaWebLidl(self, KeyWord):
        driver = webdriver.Chrome(ChromeDriverManager().install())
        urlLidl = "https://recetas.lidl.es/"
        driver.get(urlLidl)
        time.sleep(0.5)
        self.quitarCookiesLidl(driver)
        time.sleep(1)
        self.sacarInfoLidl(driver, KeyWord)
        return driver
    def conexionPaginaWebAhorraMas(self, KeyWord):

        urlAhorraMas = "https://www.ahorramas.com/buscador?q=" + KeyWord
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

    def sacarInfoLidl(self, driver, KeyWord):
        # Para entrar en el alimento que nosotros queremos
        escribirIngrediente = driver.find_element(By.CLASS_NAME, "inputField.js_mIngredientSearchGroup-input")
        escribirIngrediente.send_keys(KeyWord)
        # hay que darle un poco de tiempo para que despues de escribir seleccione el enter sino no lo ejecuta bien
        time.sleep(2)
        escribirIngrediente.send_keys(Keys.RETURN)

        time.sleep(2)
        # PARA COGER EL NOMBRE DE LA RECETA DEL PRODUCTO
        todasRecetas = driver.find_element(By.CLASS_NAME, "oRecipeFeed-resultContainer.js_oRecipeFeed-resultContainer")
        time.sleep(1)
        nombre = todasRecetas.find_elements(By.CLASS_NAME, "mRecipeTeaser-title")
        listaNombres = []
        for n in nombre:
            self.listaNombres.append(n.text)

        # PARA COGER EL TIEMPO DEL PRUDUCTO EN COCINARSE
        tiempoReceta = todasRecetas.find_elements(By.CLASS_NAME, "mTimer-time")
        listaTiempos = []
        for t in tiempoReceta:
            self.listaTiempos.append(t.text)
        time.sleep(0.5)
        # PARA COGER LA IMAGEN DEL PRODUCTO
        src = todasRecetas.find_elements(By.CLASS_NAME, "picture-img.mRecipeTeaser-image.lazyloaded")
        listaImagenes = []
        for s in src:
            self.listaImagenes.append(s.get_attribute("src"))

        # PARA COGER LA URL DE LA RECETA
        href = todasRecetas.find_elements(By.CLASS_NAME, "mRecipeTeaser-link")
        listaURL = []
        for h in href:
            self.listaURL.append(h.get_attribute("href"))

        i = 0
        for i in range(len(listaImagenes)):
            print(listaNombres[i])
            print(listaTiempos[i])
            print(listaImagenes[i])
            print(listaURL[i])
            print("-----------------------------------")
            i += 1

    def sacarInfoAhorraMas(self, soup):
        # PARA COGER EL NOMBRE DEL PRODUCTO
        bodyInformacion = soup.find_all(class_="tile-body")
        listaNombres = []
        for np in bodyInformacion:
            nombreProducto = np.find(class_="link product-name-gtm")
            self.listaNombres.append(nombreProducto.text)

        # PARA COGER EL PRECIO DEL PRODUCTO
        bodyInformacion = soup.find_all(class_="tile-body")
        listaPrecios = []
        for p in bodyInformacion:
            valorProducto = p.find(class_="value")
            # utilizamos strip para eliminar los espacios en blanco tanto delante como detras del precio
            self.listaPrecios.append(valorProducto.text.strip())
            # PARA COGER LA IMAGEN DEL PRODUCTO
            src = soup.find_all(class_="tile-image")
            listaImagenes = []
            for s in src:
                self.listaImagenes.append(s["src"])

            # PARA COGER LA URL DEL PRODUCTO
            href = soup.find_all(class_="product-pdp-link")
            listaURL = []
            for h in href:
                # concatenamos la direccion de la pagina la cual no incluye el href al usar beautifulsoup
                self.listaURL.append("https://www.ahorramas.com" + h["href"])
            # Eliminamos los elementos duplicados ya que coge cada url duplicada
            # print(len(listaURL))
            listaURLSinDuplicados = []
            for url in listaURL:
                if url not in listaURLSinDuplicados:
                    listaURLSinDuplicados.append(url)
            # print(len(listaURLSinDuplicados))

            i = 0
            for i in range(len(listaImagenes)):
                print(listaPrecios[i])
                print(listaNombres[i])
                print(listaImagenes[i])
                print(listaURLSinDuplicados[i])
                print("-----------------------------------")
                i += 1


if __name__=='__main__':
    app=QApplication(sys.argv)


    gui=Index()
    gui.showMaximized()
    gui.show()

    sys.exit(app.exec_())
