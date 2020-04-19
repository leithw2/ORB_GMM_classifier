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


    img = image_list[0]

    x=0
    for image in image_list:
        if x==0:
            des =  getDescriptors(image)
        else:
            des = np.concatenate((des, getDescriptors(image)), axis=0)
        x=x+1

    print ("los 500 descriptores de 1 imagen: ")
    print(des)

    print ("los 32 valores de 1 descriptor de la imagen 1: ")
    print (des[0])

    k = 2
    #model = GaussianMixture(n_components = k, init_params='random')
    #model.fit(des)

    filename = 'finalized_model100.sav'
    #pickle.dump(model, open(filename, 'wb'))
    model = pickle.load(open(filename, 'rb'))

    print ("Modelo entrenado")
    print (model)
    i=0
    cap = cv2.VideoCapture(0)

    prob = getPrediction(model, img)
    fresa = np.argmax(prob)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        prob = getPrediction(model, frame)
        cluster = np.argmax(prob)


        print (prob)
        if cluster == fresa:
            print("fresa")

        else:
            print("mora")

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




def getPrediction(model, image):
    des = getDescriptors(image)

    probs = model.predict_proba(des)

    sumProb = np.sum(probs, axis = 0)

    prob = [x / len(probs) for x in sumProb]
    return prob

def getDescriptors(img):
    #create orb
    orb = cv2.ORB_create(nfeatures=200)

    # find keypoints ORB
    kp = orb.detect(img,None)

    # compute descriptors ORB
    kp, des = orb.compute(img, kp)

    return(des)

if __name__ == '__main__':
    main()
