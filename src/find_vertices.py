import cv2 
import numpy as np
import math
import cutting_algos as alg
import matplotlib.pyplot as plt

def main():
    max_x = 250
    max_y = 250
    #get image
    img = cv2.imread("IMG-2333.jpg")
    
    img = mask_img(img)
    img = segment_image(img)
    img, vertices = approx_shape(img)
    # v_0, v_1 = alg.find_cut2(vertices, max_x, max_y, img)
    v_0, v_1 = alg.find_cut_naive(vertices, max_x, max_y,k=400)
    # v_0, v_1 = alg.find_cut_dfs(vertices, max_x, max_y, img)
    
    angle = get_angle(v_0, v_1)


def get_cutting_info(img, method="naive"):
    img = mask_img(img)
    img = segment_image(img)
    img, vertices = approx_shape(img)
    if method == "find_cut2":
        v_0, v_1 = alg.find_cut2(vertices, max_x, max_y, img)
    elif method == "dfs":
        v_0, v_1 = alg.find_cut_dfs(vertices, max_x, max_y, img)
    else:
        v_0, v_1 = alg.find_cut_naive(vertices, max_x, max_y,k=400)
    angle = get_angle(v_0, v_1)
    return v_0, angle

def get_angle(v_0, v_1):
    # double check
    v = v_0 if v_0[1] < v_1[1] else v_1  # x is element 0, y is element 1
    
    theta = math.tan(v[1], v[0])
    if theta < 0:
        theta += 180
    return theta


def mask_img(img):
    """ mask the image """
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.circle(mask, (1450, 1950), 1400, 255,-1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    

    return masked

def segment_image(img):
    """ Removes the plate's background and leaves the food""" 
    lower_pink = (110, 0, 0)
    upper_pink = (230, 255, 255)
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask = cv2.bitwise_not(cv2.inRange(hsv_img, lower_pink, upper_pink))
    result = cv2.bitwise_and(img, img, mask=mask)
    black_pixels = np.where(
        (result[:, :, 0] == 0) & 
        (result[:, :, 1] == 0) & 
        (result[:, :, 2] == 0)
    )

    # set those pixels to white
    result[black_pixels] = [255, 255, 255]
    kernel = np.ones((5,5), np.uint8)
    result = cv2.dilate(result, kernel, iterations=4)
    
    return result 


def approx_shape(img):
    # Convert to greyscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert to binary image by thresholding
    _, threshold = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY_INV)
    # Find the contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print (len(contours))
    # For each contour approximate the curve and
    # detect the shapes.
    vertices = 0
    best_area = 100
    best_approx=-1
    tmp = img.copy()
    for cnt in contours :
        area = cv2.contourArea(cnt)
    
        # Shortlisting the regions based on there area.
        # TODO: select the set of approx that gives the biggest area
        if area > best_area: 
            approx = cv2.approxPolyDP(cnt, 
                                    0.01 * cv2.arcLength(cnt, True), True)
            
            best_area = area 
            best_approx = approx
            cv2.drawContours(tmp, [best_approx], 0, (0, 0, 255), 3)

            vertices = best_approx
            for i in range (len(approx)):
                app = approx[i].ravel()
                cv2.circle(tmp, (int(app[0]), int(app[1])), 3, 30, 20, -1)
                cv2.putText(tmp, f"vertex {i}",(int(app[0]), int(app[1])),  
                cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0) )
            
    
    cv2.imshow("approximated shape", tmp)
    #cv2.waitKey()
    #print (vertices)
    vertices = vertices.reshape(len(vertices),2)
    #print (vertices)
    
    return img, vertices


""" 

https://stackoverflow.com/questions/7263621/how-to-find-corners-on-a-image-using-opencv
https://stackoverflow.com/questions/26561220/find-vertex-from-a-object-by-using-vertex-detection
"""

if __name__ == "__main__":
    main()

