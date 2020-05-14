#!/usr/bin/env python3
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from numpy import hstack
from numpy.random import normal
from sklearn.mixture import GaussianMixture
from sklearn import svm
from sklearn import datasets
import pickle


from os import getcwd

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QFileDialog

# ===================== CLASE QLabelClickable ======================

class QLabelClickable(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super(QLabelClickable, self).__init__(parent)

    def mousePressEvent(self, event):
        self.clicked.emit()


# ====================== CLASE mostrarImagen =======================

class mostrarImagen(QDialog):
    def __init__(self, parent=None):
        super(mostrarImagen, self).__init__(parent)

        self.setWindowTitle("RECONOCIMIENTO OBJETO")
        self.setWindowIcon(QIcon("icono.png"))
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.MSWindowsFixedSizeDialogHint)
        self.setFixedSize(1000, 600)

        self.initUI()

    def initUI(self):
        self.imagen = []
      # ==================== WIDGET QLABEL OF IMAGE =======================

        self.labelImagen = QLabelClickable(self)
        self.labelImagen.setGeometry(250, 15, 700, 400)
        self.labelImagen.setToolTip("Imagen")
        self.labelImagen.setCursor(Qt.PointingHandCursor)
        self.labelImagen.setStyleSheet("QLabel {background-color: white; border: 1px solid "
                                       "#01DFD7; border-radius: 5px;}")
        self.labelImagen.setAlignment(Qt.AlignCenter)

      #===================WIDGET QLABEL OF FRUIT NAMES====================
        self.labelFruit1 = QLabelClickable(self)
        self.labelFruit1.setGeometry(400, 470, 200, 35)
        self.labelFruit1.setToolTip("Fruitname")
        self.labelFruit1.setCursor(Qt.PointingHandCursor)
        self.labelFruit1.setStyleSheet("QLabel {background-color: white; border: 1px solid "
                                       "#01DFD7; border-radius: 5px;}")
        self.labelFruit1.setAlignment(Qt.AlignCenter)


        self.labelFruit2 = QLabelClickable(self)
        self.labelFruit2.setGeometry(400, 520, 200, 35)
        self.labelFruit2.setToolTip("Fruitname")
        self.labelFruit2.setCursor(Qt.PointingHandCursor)
        self.labelFruit2.setStyleSheet("QLabel {background-color: white; border: 1px solid "
                                       "#01DFD7; border-radius: 5px;}")
        self.labelFruit2.setAlignment(Qt.AlignCenter)

      #===================WIDGET QLABEL OF PROBABILITY====================
        self.labelPercent1 = QLabelClickable(self)
        self.labelPercent1.setGeometry(650, 470, 100, 35)
        self.labelPercent1.setToolTip("Prob")
        self.labelPercent1.setCursor(Qt.PointingHandCursor)
        self.labelPercent1.setStyleSheet("QLabel {background-color: white; border: 1px solid "
                                       "#01DFD7; border-radius: 5px;}")
        self.labelPercent1.setAlignment(Qt.AlignCenter)

        self.labelPercent2 = QLabelClickable(self)
        self.labelPercent2.setGeometry(650, 520, 100, 35)
        self.labelPercent2.setToolTip("Prob")
        self.labelPercent2.setCursor(Qt.PointingHandCursor)
        self.labelPercent2.setStyleSheet("QLabel {background-color: white; border: 1px solid "
                                       "#01DFD7; border-radius: 5px;}")
        self.labelPercent2.setAlignment(Qt.AlignCenter)

      #=============== LABEL OF OBJECT NAMES================
        self.labelName1 = QLabel("Objeto", self)
        self.labelName1.setGeometry(450, 420, 100, 35)
        self.labelName1.setToolTip("Objeto")
        self.labelName1.setCursor(Qt.PointingHandCursor)
        self.labelName1.setStyleSheet("QLabel {background-color: --; border: 1px solid "
                                       "#01DFD7; border-radius: 5px;}")
        self.labelName1.setAlignment(Qt.AlignCenter)


        self.labelName2 = QLabel("Probabilidad, %", self)
        self.labelName2.setGeometry(600, 420, 200, 35)
        self.labelName2.setToolTip("Probabilidad")
        self.labelName2.setCursor(Qt.PointingHandCursor)
        self.labelName2.setStyleSheet("QLabel {background-color: --; border: 1px solid "
                                       "#01DFD7; border-radius: 5px;}")
        self.labelName2.setAlignment(Qt.AlignCenter)
      # ================= WIDGETS QPUSHBUTTON ====================

        buttonSeleccionar = QPushButton("Seleccionar", self)
        buttonSeleccionar.setToolTip("Seleccionar imagen")
        buttonSeleccionar.setCursor(Qt.PointingHandCursor)
        buttonSeleccionar.setGeometry(100, 30, 120, 25)

        buttonAnalizar = QPushButton("Analizar", self)
        buttonAnalizar.setToolTip("Analizar imagen")
        buttonAnalizar.setCursor(Qt.PointingHandCursor)
        buttonAnalizar.setGeometry(100, 100, 120, 25)

        buttonEliminar = QPushButton("Limpiar", self)
        buttonEliminar.setToolTip("Eliminar imagen")
        buttonEliminar.setCursor(Qt.PointingHandCursor)
        buttonEliminar.setGeometry(100, 200, 120, 25)

      # ===================== EVENTO QLABEL ======================

      # Llamar función al hacer clic sobre el label
        #self.labelImagen.clicked.connect(self.seleccionarImagen)

      # ================== EVENTOS QPUSHBUTTON ===================

        buttonSeleccionar.clicked.connect(self.seleccionarImagen)
        buttonAnalizar.clicked.connect(self.run)
        buttonEliminar.clicked.connect(lambda: self.labelImagen.clear())


  # ======================= FUNCIONES ============================

    def seleccionarImagen(self):

        imagen, extension = QFileDialog.getOpenFileName(self, "Seleccionar imagen", getcwd(),
                                                        "Archivos de imagen (*.png *.jpg)",
                                                        options=QFileDialog.Options())

        if imagen:
            # Adaptar imagen
            pixmapImagen = QPixmap(imagen).scaled(650,550, Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)

            # Mostrar imagen
            self.labelImagen.setPixmap(pixmapImagen)
            self.imagen=cv2.imread(imagen,cv2.COLOR_RGB2BGR)

        self.labelFruit1.setText('')
        self.labelPercent1.setText("")
        self.labelFruit2.setText('')
        self.labelPercent2.setText("")

    def run(self):
        image_list = []
        test_list = []

        #Cargar modelo----------------
        img=self.imagen
        image_to_compare = cv2.imread("train_images/f1.jpg",cv2.COLOR_RGB2BGR)
        print(img)

        filename = 'finalized_model100.sav'
        model = pickle.load(open(filename, 'rb'))

        print ("Modelo entrenado")
        print (model)

        #Calculo de probabilidad sabiendo que es fresa
        prob = self.getPrediction(model, img)
        prob2 = self.getPrediction(model, image_to_compare)
        #Cluster con valor max
        fresa = np.argmax(prob)
        thing = np.argmax(prob2)
        print(fresa)

        self.labelFruit1.setText('Fresa')
        string = "" + str(round(prob[0],3))
        self.labelPercent1.setText(string)

        self.labelFruit2.setText('Mora')
        string = "" + str(round(prob[1],3))
        self.labelPercent2.setText(string)
        # if fresa == 0:
        #     self.labelFruit1.setText('Fresa')
        #     string = "" + str(round(prob2[0],3))
        #     self.labelPercent1.setText(string)
        #     self.labelFruit2.setText('Mora')
        #     string = "" + str(round(prob2[1],3))
        #     self.labelPercent2.setText(string)

        # else:
        #
        #     self.labelFruit1.setText('Fresa')
        #     string = "" + str(round(prob2[1],3))
        #     self.labelPercent1.setText(string)
        #     self.labelFruit2.setText('Mora')
        #     string = "" + str(round(prob2[0],3))
        #     self.labelPercent2.setText(string)

    def getPrediction(self,model, image):

        #Descriptores de la imagen
        des = self.getDescriptors(image)

        #Probabilidades de cada descriptor
        probs = model.predict_proba(des)
        #print(probs)
        #Promediar probabilidades
        sumProb = np.sum(probs, axis = 0)
        prob = [x / len(probs) for x in sumProb]
        return prob

    def getDescriptors(self,img):
        #create orb nfeatures= numero de descriptores por imagen
        orb = cv2.ORB_create(nfeatures=200)

        # find keypoints ORB
        kp = orb.detect(img,None)

        # compute descriptors ORB
        kp, des = orb.compute(img, kp)

        return(des)
  # ================================================================

if __name__ == '__main__':

    import sys

    aplicacion = QApplication(sys.argv)

    ventana = mostrarImagen()
    ventana.show()

    sys.exit(aplicacion.exec_())
