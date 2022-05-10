#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image 
from cut_finder.srv import *


def callback(data):
    rospy.wait_for_service("find_cut")
    fxn = rospy.ServiceProxy('find_cut', find_cut)
    test = fxn(data)
    print ("hello")
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('test_cut_service', anonymous=True)

    rospy.Subscriber("/camera/color/image_raw", Image, callback)


    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()