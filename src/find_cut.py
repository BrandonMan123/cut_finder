#!/usr/bin/env python


from __future__ import print_function 

import rospy
import find_vertices as finder
import tf
from cut_finder.srv import find_cut, find_cutResponse
from cv_bridge import CvBridge
import image_geometry
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import numpy as np
"""
TODO: 
1. Find cv image type
2. Find how to return message to main
translation: [0.18710371269311812, -0.43625319322764927, 0.6480278593440019]

d * v gives you the position of the object relative to the camera
 """

def pixel_to_coordinate(x, y):
    camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo, timeout=None)
    camera_matrix = np.asarray(camera_info.K).reshape(3,3)
    cam_fx = camera_matrix[0,0]
    cam_fy = camera_matrix[1,1]
    cam_cx = camera_matrix[0,2]
    cam_cy = camera_matrix[1,2]
    tz=0.45
    tx = (tz/cam_fx) * (x-cam_cx)
    ty = (tz/cam_fy) * (y-cam_cy)
    return tx, ty

def find_vertices(req):
    # Convert img to cv2 image
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(req.img, desired_encoding="bgr8")

    #coord, angle = finder.get_cutting_info(img)
    coord = (621, 240)
    angle = 10
    
    listener = tf.TransformListener()

    x, y = pixel_to_coordinate(coord[0], coord[1])
    print ("true x and y", x, y)
    print ("pix to coord",pixel_to_coordinate(coord[0], coord[1]))
    trans = 0
    while trans == 0:
        try:
            trans, rot = listener.lookupTransform("/base_link", 
            "/camera_color_optical_frame", rospy.Time(0))
            
        except (tf.LookupException, tf.ConnectivityException, 
        tf.ExtrapolationException):
            continue
    print ("coord is ", coord, "translation is ", trans)
    ret_coord = (trans[0]+x*-1+0.11, y-0.73)
    finder.display_cut(img, coord)
    # ret_coord = coord
    # angle +=  rot
    print ("final x coord", ret_coord[0])
    print ("final y coord", ret_coord[1])
    return [ret_coord[0], ret_coord[1], angle]

def find_vertex_server():
    rospy.init_node('find_cut_server')
    print ('starting server')
    s = rospy.Service('find_cut', find_cut, find_vertices)
    rospy.spin()

if __name__ == "__main__":
    find_vertex_server()
