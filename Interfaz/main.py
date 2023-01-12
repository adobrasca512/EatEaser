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
        self.btn_knn = QPushButton('K-NN')
        self.btn_rf = QPushButton('Random-Forest')
        self.btn_rn = QPushButton('Red Neuronal')
        self.btnalgoritmo = QPushButton()
        # estilizamos botones
        self.btn_knn.setFixedSize(QtCore.QSize(400, 80))
        self.btn_rf.setFixedSize(QtCore.QSize(400, 80))
        self.btn_rn.setFixedSize(QtCore.QSize(400, 80))
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
        self.btn_knn.setStyleSheet(sbtnalgoritmo)
        self.btn_rf.setStyleSheet(sbtnalgoritmo)
        self.btn_rn.setStyleSheet(sbtnalgoritmo)
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
        self.algoritmolayout.addWidget(self.btn_knn, 1, 0, 2, 1)
        self.algoritmolayout.addWidget(self.btn_rf, 1, 1, 2, 1)
        self.algoritmolayout.addWidget(self.btn_rn, 1, 2, 2, 1)
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
        self.btn_knn.clicked.connect(
            lambda: self.informacion('Algoritmo K-NN', 'Este algoritmo hace esto y esto y esto'))
        self.btn_rn.clicked.connect(
            lambda: self.informacion('Algoritmo Red Neuronal', 'Este algoritmo hace esto y esto y esto'))
        self.btn_rf.clicked.connect(
            lambda: self.informacion('Algoritmo Random Forest', 'Este algoritmo hace esto y esto y esto'))
        self.btnalgoritmo.clicked.connect(self.vista_previa)
        self.nuevo.clicked.connect(self.aniadir_categoria)
        self.retorno.clicked.connect(self.volver)
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
        self.path_btn.clicked.connect(self.recuperarRutaModeloEntrenado)
        self.btn_guardar.clicked.connect(self.recuperarModeloEntrenado)
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


    def recuperarRutaModeloEntrenado(self):
        r = QFileDialog.getOpenFileName(parent=None, caption='Select Directory', directory=os.getcwd(), filter='Pickle files (*.pkl)')
        self.varableRutaModeloEntrenado=r[0]
    def recuperarModeloEntrenado(self):
        if(self.varableRutaModeloEntrenado!=""):
            print('--------------------------')
            modelo_entrenado = joblib.load(self.varableRutaModeloEntrenado)
            #print(modelo_entrenado)
            #print(modelo_entrenado.score(x_train, y_train))

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




import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

import time


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
                grid_titulo.addWidget(QLabel('Titlo de video'))
                frame_grid.addLayout(grid_boton)
                ver=QPushButton('Ver texto')
                ver.setStyleSheet('background-color:black;color:white;font-family:Californian FB;font-size:16px;border-radius:20px;')
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
    app=QApplication(sys.argv)

    gui=Download()
    gui.show()
    gui.showMaximized()


    sys.exit(app.exec_())



