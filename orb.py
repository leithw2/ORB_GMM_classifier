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
from matplotlib.colors import LogNorm
import pickle


def main():
    image_list = []
    test_list = []

    #Reading images---------------
    for filename in sorted(glob.glob('train_images/*')):

        print (filename)
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #Resize images---------------
        #width = im.shape[1]
        #height = int(300.0/width * im.shape[0])
        #im = cv2.resize(im, (300,height), interpolation = cv2.INTER_AREA)
        image_list.append(im)


    for filename in sorted(glob.glob('test_images/*')):

        print (filename)
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #Resize images---------------
        #width = im.shape[1]
        #height = int(300.0/width * im.shape[0])
        #im = cv2.resize(im, (300,height), interpolation = cv2.INTER_AREA)
        test_list.append(im)

    #Imagen de prueba fresa
    img = image_list[0]

    #Concatenar los descriptores
    x=0
    for image in image_list:
        if x==0:
            #guarda el primer descriptor
            des =  getDescriptors(image)
        else:
            #concatena los otros descriptores
            des = np.concatenate((des, getDescriptors(image)), axis=0)
        x=x+1

    k = 2

    #Guardar modelo---------------
    #model = GaussianMixture(n_components = k, init_params='random')
    #model.fit(des)
    #filename = 'finalized_model.sav'
    #pickle.dump(model, open(filename, 'wb'))

    #Cargar modelo----------------
    filename = 'finalized_model100.sav'
    model = pickle.load(open(filename, 'rb'))

    print ("Modelo entrenado")
    print (model)

    #Calculo de probabilidad sabiendo que es fresa
    prob = getPrediction(model, img)
    #Cluster con valor max
    fresa = np.argmax(prob)

    for image in test_list:

        #Calculo de probabilidad imagenes en test_list
        prob = getPrediction(model, image)
        #Cluster con valor max
        cluster = np.argmax(prob)

        if cluster == fresa:
            print("fresa")

        else:
            print("mora")


def getPrediction(model, image):

    #Descriptores de la imagen
    des = getDescriptors(image)

    #Probabilidades de cada descriptor
    probs = model.predict_proba(des)

    #Promediar probabilidades
    sumProb = np.sum(probs, axis = 0)
    prob = [x / len(probs) for x in sumProb]
    return prob

def getDescriptors(img):
    #create orb nfeatures= numero de descriptores por imagen
    orb = cv2.ORB_create(nfeatures=200)

    # find keypoints ORB
    kp = orb.detect(img,None)

    # compute descriptors ORB
    kp, des = orb.compute(img, kp)

    return(des)

if __name__ == '__main__':
    main()
