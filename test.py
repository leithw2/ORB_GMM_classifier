#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


from cv_bridge import CvBridge, CvBridgeError
class image_converter:

  def __init__(self):
    self.cmc_vel_pub = rospy.Publisher("/turtle1/cmd_vel",Twist)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cv_camera/image_raw",Image,self.callback)
    self.vel_msg = Twist()



  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    self.centerx = int(cols/2)

    frame = cv_image

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_red = np.array([0,200,50])
    upper_red = np.array([50,255,255])

    # Threshold the HSV image to get only Red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #cv2.imshow('HSV',hsv)
    cv2.imshow('res',res)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    img, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img = frame
    img = cv2.drawContours(frame,contours, -1, (255,255,0), 3)
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area >= 500:
            x,y,w,h = cv2.boundingRect(cnt)
            #img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            print (area)
            print (x)
            if area >= 5500:
                self.vel_msg.linear.x = -0.1
                self.vel_msg.linear.y = 0.0
                self.vel_msg.angular.x = 0
                self.vel_msg.angular.y = 0
                self.vel_msg.angular.z = 0
                self.cmc_vel_pub.publish(self.vel_msg)
                print ("BackWard")

            if area < 1000:
                self.vel_msg.linear.x = 0.1
                self.vel_msg.linear.y = 0.0
                self.vel_msg.angular.x = 0
                self.vel_msg.angular.y = 0
                self.vel_msg.angular.z = 0
                self.cmc_vel_pub.publish(self.vel_msg)
                print ("FordWard")

            if x <= self.centerx - 200 :
                self.vel_msg.linear.x = 0
                self.vel_msg.linear.y = 0.0
                self.vel_msg.angular.x = 0
                self.vel_msg.angular.y = 0
                self.vel_msg.angular.z = 0.5
                self.cmc_vel_pub.publish(self.vel_msg)
                print ("Left")

            if x > self.centerx + 200 :
                self.vel_msg.linear.x = 0
                self.vel_msg.linear.y = 0.0
                self.vel_msg.angular.x = 0
                self.vel_msg.angular.y = 0
                self.vel_msg.angular.z = -0.5
                self.cmc_vel_pub.publish(self.vel_msg)
                print ("Right")
            break
        else:
            img=frame

    cv2.imshow('Contornos',img)
    cv2.waitKey(2)

    try:
      pass
      #self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
