#!/usr/bin/env python


from __future__ import print_function 

import rospy
import find_vertices as finder
import tf
from cut_finder.srv import find_cut, find_cutResponse


"""
TODO: 
1. Find cv image type
2. Find how to return message to main

 """
def find_vertices(req):
    angle, coord = finder.find_vertices(req)
    listener = tf.TransformListener()
    trans = 0
    while trans == 0:
        try:
            trans, rot = listener.lookupTransform("/base_link", 
            "/camera_color_optical_frame", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, 
        tf.ExtrapolationException):
            continue
    ret_coord = (coord[0]+trans[0], coord[1]+trans[1])
    angle +=  rot
    return [ret_coord, angle]

def find_vertex_server():
    rospy.init_node('find_cut_server')
    s = rospy.Service('find_cut', find_cut, find_vertices)
    rospy.spin()

if __name__ == "__main__":
    find_vertex_server()
