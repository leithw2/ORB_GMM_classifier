import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob


def main():
    image_list = []

    for filename in sorted(glob.glob('train_images/*')):
        print filename
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
         # percent of original size
        width = im.shape[1]
        height = int(300.0/width * im.shape[0])
        print height
        im = cv2.resize(im, (300,height), interpolation = cv2.INTER_AREA)
        # h = 300
        # w = 300
        # y = im.shape[0]/2 - h/2
        # x = im.shape[1]/2 - w/2
        # im = im[y:y+h, x:x+w]
        # plt.imshow(im)
        # plt.show()
        image_list.append(im)


    img = image_list[1]

    img2 = image_list[2]
    #img2 = cv2.imread('nofresa.jpg')
    # Initiate STAR detector


    color = ('b','g','r')

    plt.subplot(2,2,3)
    for i, c in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])

    plt.subplot(2,2,4)

    for i, c in enumerate(color):
        hist = cv2.calcHist([img2], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])

    #Printing-----

    plt.subplot(2,2,1)
    plt.imshow(img)

    plt.subplot(2,2,2)
    plt.imshow(img2)


    plt.show()
    #run(img, img2)

    multiRun(image_list)

def run(img, img2):
    orb = cv2.ORB_create(nfeatures=2000)

    # find the keypoints with ORB
    kp = orb.detect(img,None)
    kp2 = orb.detect(img2,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    kp2, des2 = orb.compute(img2, kp2)

    # crea BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des,des2)

    matches = sorted(matches, key = lambda x:x.distance)

    # Dibuja las primeras 30 coincidencias.
    img3 = cv2.drawMatches(img,kp,img2,kp2,matches[:30],None, flags=2)
    suma = 0
    for match in matches:
        print match.distance
        suma = suma + match.distance

    media = suma / len(matches)
    print (media)
    print (len(des))
    print (len(des2))
    print (len(matches))

    plt.imshow(img3),plt.show()

def multiRun(list):

    orb = cv2.ORB_create(nfeatures=2000)

    first = list[0]
    x=0
    for image in list:

        # find the keypoints with ORB
        kp = orb.detect(first,None)
        kp2 = orb.detect(image,None)

        # compute the descriptors with ORB
        kp, des = orb.compute(first, kp)
        kp2, des2 = orb.compute(image, kp2)

        # crea BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des,des2)

        matches = sorted(matches, key = lambda x:x.distance)

    # Dibuja las primeras 30 coincidencias.
        img3 = cv2.drawMatches(first,kp,image,kp2,matches[:30],None, flags=2)
        suma = 0
        for match in matches:
            print match.distance
            suma = suma + match.distance

        media = suma / len(matches)
        print (media)
        print ("des1 =" + str(len(des)))
        print ("des2 =" + str(len(des2)))
        print ("matches =" + str(len(matches)))
        print ("media =" + str(media))

        plt.figure(figsize=(15,10))
        plt.imshow(img3)
        plt.title('Matches = ' + str(len(matches)))
        plt.savefig("resultado" + str(x) + ".png")
        x=x+1
        img3=None
    plt.show()

if __name__ == '__main__':
    main()
