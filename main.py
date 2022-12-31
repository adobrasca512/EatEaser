import os
import sys
import shutil
from PyQt5 import QtCore, QtWidgets, QtGui, Qt
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase, QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QGridLayout, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, \
    QWidget, QSizePolicy, QComboBox, QLayout, QFormLayout, QLineEdit, QButtonGroup, QRadioButton, QCheckBox, QFileDialog


class Index(QWidget):
    def __init__(self):
        super().__init__()
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
    def openTrain(self):
        self.gui = Train()
        self.gui.show()
        self.gui.showMaximized()
        self.close()
class Train(QWidget):
    def __init__(self):
        super().__init__()
        #variables globales
        self.seleccionados = []
        self.checkboxes = []
        self.total_archivos=0
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
        self.cbcategoria.addItems(os.listdir('C:/Users/adobr/Downloads/EatEaser-pruebasRober (1)/EatEaser-pruebasRober/recetastextos/'))
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


        carpetas=''

        for i in self.seleccionados:

            size = len(os.listdir('C:/Users/adobr/Downloads/EatEaser-pruebasRober (1)/EatEaser-pruebasRober/recetastextos/' + i))
            print(self.total_archivos)
            self.total_archivos = size + self.total_archivos
            texto=i + ': ' + str(size) + ' archivos\n'

            carpetas=carpetas+'\n'+texto
            print(self.total_archivos)


        # le añado todos los que esten en listbox
        self.vista.setText(carpetas+'\n'+'TOTAL: ' + ': ' + str(self.total_archivos) + ' archivos\n')


    def aniadir_boton(self):
        self.add=QCheckBox(self.cbcategoria.currentText())
        self.seleccionados.append(str(self.cbcategoria.currentText()))
        self.add.setStyleSheet('QPushButton{'+'background-color:transparent;border-radius:12px;border:2px solid black;font-family:"Bahnschrift Light";font-size:16px;letter-spacing:3px;}QPushButton:hover{background-color:'
                                              'black;color:white}')

        self.seleccionlayout.addWidget(self.add)
        self.checkboxes.append(self.add)
        print(self.seleccionados)

    def eliminar_boton(self):

        i=0
        for c in self.checkboxes:
            if c.isChecked()==True:

                c.deleteLater()
                self.seleccionados.pop(i)

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
            destination = 'C:/Users/adobr/Downloads/EatEaser-pruebasRober (1)/EatEaser-pruebasRober/recetastextos/'+ultimo+'/' + file_name
            print('se va al destino',destination)
            #si existe el archivo de source lo movemos al destino

            if os.path.exists('C:/Users/adobr/Downloads/EatEaser-pruebasRober (1)/EatEaser-pruebasRober/recetastextos/'+ultimo)==False:
                os.makedirs('C:/Users/adobr/Downloads/EatEaser-pruebasRober (1)/EatEaser-pruebasRober/recetastextos/'+ultimo)
                shutil.move(source, destination)
                print('Moved:', file_name)
            else:
                #aqui va a haber un error
                shutil.move(source, destination)
                print('Moved:', file_name)



        #actualizamos el combobox

        self.cbcategoria.clear()

        self.cbcategoria.addItems(os.listdir('C:/Users/adobr/Downloads/EatEaser-pruebasRober (1)/EatEaser-pruebasRober/recetastextos/'))

    def aniadir_directorio(self):
        r=QFileDialog.getExistingDirectory(self, "Select Directory",directory=os.getcwd())
        print(r)
        self.formguardar.setPlaceholderText(r)
    def volver(self):
        self.gui = Index()
        self.gui.show()
        self.gui.showMaximized()
        self.close()





if __name__=='__main__':
    app=QApplication(sys.argv)


    gui=Index()
    gui.showMaximized()
    gui.show()

    sys.exit(app.exec_())
