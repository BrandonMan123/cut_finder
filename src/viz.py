import cv2 
import numpy as np
import math
import cutting_algos as alg
import matplotlib.pyplot as plt
import find_vertices as vert

def main():
    max_x = 1250
    max_y = 1250
    #get image
    img = cv2.imread("IMG-2333.jpg")
    
    img = vert.mask_img(img)
    img = vert.segment_image(img)
    
    for i in range (0, 5):
        img, vertices = vert.approx_shape(img)
        #v_0, v_1 = alg.find_cut_naive(vertices, max_x, max_y, img, 300)
        v_0, v_1 = alg.find_cut2(vertices, max_x, max_y, img)
        #v_0, v_1 = alg.find_cut_dfs(vertices, max_x, max_y, img)
        print (v_0)
        tmp = img.copy()
        cv2.circle(tmp, v_0, 3, 100, 60, -1)
        cv2.putText(tmp, "cut start", v_0, cv2.FONT_HERSHEY_SIMPLEX,  2, (255, 1, 0), thickness=2)
        cv2.circle(tmp, v_1, 3, 200, 45, -1)
        cv2.putText(tmp, "cut end", v_1, cv2.FONT_HERSHEY_SIMPLEX,  2, (255, 1, 0), thickness=2)

        crop_img(img, v_0, v_1)
        cv2.imshow("cut", tmp)
        cv2.waitKey()
    

def crop_img(img, v_0, v_1, k=1000):
    # calculate slope
    print (v_0, v_1)
    m = (v_0[1] - v_1[1])/(v_0[0] - v_1[0])
    print ("slope is", m)
    k = k * -1 
    if m == 0:
        m = k
    v_0_ext = (v_0[0] + k, int(-k/m + v_0[1]))
    v_1_ext = (v_1[0] + k, int(-k/m + v_1[1]))
    vertices = np.array([
        [v_0[0], v_0[1]],
        [v_1[0], v_1[1]],
        [v_1_ext[0], v_1_ext[1]],
        [v_0_ext[0], v_0_ext[1]]
    ])

    #vertices = np.array([[50,50], [50,150], [150,150], [150,50]])

    cv2.fillPoly(img, pts=[vertices], color =(255,255,255))
    return img
    


if __name__ == "__main__":
    main()