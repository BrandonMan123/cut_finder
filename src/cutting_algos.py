import numpy as np
import cv2


"""
Naive algorithm:
Initialize v_0 to be the leftmost vertex
Find smallest y from list of vertices and the largest, call them y_min, y_max
Add k to x_0, call it x_cut
return (x_cut, y_min), (x_cut, y_max)
Mark those two vertices as cutting points
 """
def find_cut_naive(points, max_x, max_y, img, k=80):
    min_x = np.min(points[:,0])
    x_cut = min_x + k 
    sorted_y = np.sort(points[:,1])
    y_max = int(sorted_y[-1])
    y_min = int(sorted_y[0])

    return (x_cut, y_min), (x_cut, y_max)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def viz_img(img, v_0, v_1, v_2, msg = "debug"):
    tmp = img.copy()
    cv2.circle(tmp, v_0, 3, 100, 60, -1)
    cv2.putText(tmp, "vertex start", v_0, cv2.FONT_HERSHEY_SIMPLEX,  2, (255, 1, 0), thickness=2)
    cv2.circle(tmp, v_1, 3, 200, 45, -1)
    cv2.putText(tmp, "vertex 1", v_1, cv2.FONT_HERSHEY_SIMPLEX,  2, (255, 1, 0), thickness=2)
    cv2.circle(tmp, v_2, 3, 200, 45, -1)
    cv2.putText(tmp, "vertex 2", v_2, cv2.FONT_HERSHEY_SIMPLEX,  2, (255, 1, 0), thickness=2)
    cv2.imshow(msg, tmp)
    cv2.waitKey()

"""
Algorithm 2:
get the left most point, call it v_0.
Get the two closest points to v_0, call them v_1 and v_2


 """
def find_cut2(vertices, max_x, max_y, img):
    graph = construct_graph(vertices)
    v_0 = min(graph) #selects vertex with lowest x value
    v_1 = graph[v_0][0]

    for i in graph[v_0]:
        if i == v_1: 
            continue
        v_2 = i

    #viz_img(img, v_0, v_1, v_2, "init")
    
    dist = get_dist(v_1, v_2)
    if dist > max_x or dist > max_y:
        return get_base(v_0, v_1, v_2, dist, max_x, max_y)
    else:
        return v_1, v_2

    
    return v_start, v_end


def get_base(v_0, v_1, v_2, dist, max_x, max_y):
    i = 0
    while dist > max_x or dist > max_y:
        if i%2 == 0:
            v_2 = get_midpoint(v_0,v_2)
        else:
            v_1 = get_midpoint(v_1,v_0)
        i+= 1
        dist = get_dist(v_1, v_2)
    
    
    return int_tuple(v_1), int_tuple(v_2)


"""
Algorithm 2: dfs

Input: graph of the shape
sort list from smallest x coord to largest x coord.
intiialize v_0 to be the vertex with the smallest x coord. 
initialize best_vertex to be (0,0)
initialize v_1 to be empty
initialize v_2 to be empty
if node is not visited:
    if v_1 is empty:
        v_1 = node
        dfs(v_1, best_vertex, vertex_lst)
    v_2 = node
    dist = ||v_0 - v_2||_inf
    if dist < max x and dist < max y:
        best vertex = v_2
        dfs(v_2)
    else:
        if best vertex != (0,0)
            return best vertex
        i = 0
        while dist > max x or dist > max y:
            if i%2 == 0:
                v_2 = (v_1+v_2)/2
            else:
                v_0 = (v_1+v_0)/2
            i+= 1
            dist = ||v_0 - v_2||_inf
        return v_2
if node is visited:
    if v_2 is empty: # we are at timestep 1 of the algorithm
        dfs(v_1)
    else if node == v_0: #we walked through the entire shape
        return best vertex

 """
    
def find_cut_dfs(vertices, max_x, max_y, img):
    graph = construct_graph(vertices)
    v_0 = min(graph) #selects vertex with lowest x value
    
    v_1 = graph[v_0][0]

    for i in graph[v_1]:
        if i == v_0: 
            continue
        v_2 = i
    
    best_vertex = None
    visited = [v_0, v_1]
    #print ("graph: ", graph)
    #print (v_0, v_1, v_2)
    #viz_img(img, v_0, v_1, v_2, "init")

    def dfs(visited, graph, curr, v_0, v_1, v_2, best_vertex, img):
        #viz_img(img, v_0, v_1, v_2)

        if curr not in visited:
            # print (curr, "has not been visited. visited contains", visited)
            visited.append(curr)
            
            # print ("v_0 is", v_0,"v_1 is", v_1,"v_2 is", v_2)
            dist = get_dist(v_0, v_2)
            if dist < max_x and dist < max_y:
                
                # print ("best vertex is now", v_2, "previous best vertex was", best_vertex)
                best_vertex = v_2 
                for neighbour in graph[curr]:
                    if v_0 in graph[curr] and len(visited) > 2: # travelled graph
                        return v_0, v_1
                    # print ("visitng", neighbour, "visited nodes include", visited)
                    return dfs(visited, graph, neighbour, v_0, v_2, neighbour, best_vertex, img)
            else:
                if best_vertex is not None:
                    # print ("found optimal vertex")
                    return v_0, best_vertex
                i=0
                while dist > max_x or dist > max_y:
                    if i%2 == 0:
                        v_2 = get_midpoint(v_1,v_2)
                    else:
                        v_0 = get_midpoint(v_1,v_0)
                    i+= 1
                    dist = get_dist(v_0, v_2)
                # print ("found something")
                return v_0, v_2
        
    
    v_s, v_e = dfs(visited, graph, v_2, v_0, v_1, v_2, best_vertex, img)
    return int_tuple(v_s), int_tuple(v_e)


def to_tuple(arr):
    return (arr[0], arr[1]) 

def to_arr(tup):
    return np.array([tup[0], tup[1]], dtype=np.int32)

def construct_graph(vertices):
    """ Assuming vertices in order, construct a graph connecting 
    each vertex"""
    graph = {}
    for i in range (len(vertices)):
        v = to_tuple(vertices[i])
        v_l = to_tuple(vertices[i-1])
        v_r = to_tuple(vertices[(i+1) % len(vertices)])
        graph[v] = [v_l, v_r]
    return graph


def get_midpoint(v_1,v_2):
    midpoint = (to_arr(v_1)+to_arr(v_2))/2
    return to_tuple(midpoint)
    
def get_dist(v_0, v_2):
    return np.linalg.norm(to_arr(v_0)-to_arr(v_2), np.inf)

def int_tuple(v):
    y = list(v)
    y[0] = int(y[0])
    y[1] = int(y[1])
    return y

if __name__ =="__main__":
    vertices = np.array([[1,2],[2,3],[3,4],[5,6]])
    







""" 
Algorithm 1:

- Assumes that the shape is concave

Input: graph of the shape
sort list from smallest x coord to largest x coord.
intiialize v_0 to be the vertex with the smallest x coord. 
initialize best_vertex to be (0,0)
randomly pick either neighbour
for v_i in vertex:
    draw a line from v_0 to v_i
        if (v_0x-v_ix, v_0y-v_iy) > (x_max, y_max):
            if best_vertex is (0,0):
                connect the midpoint of v_i to v_0
            else: 
                return (v_0, best_vertex)
            connect the vertex of the midpoint and try again
        else: 
            try v_i+1
    return (v_0, v)


"""
def find_cut(points, max_x, max_y):
    """takes in list of vertices and finds a cut on the shape. returns two points
    which specifies the start and end of the cut
      """ 
    
    # handle adjacent vertices
    def find_points(x_0, x_1, x_lst, best_vertex):
        dist = np.linalg.norm(x_0-x_1, np.inf)
        if dist > max_x or dist > max_y:
            if best_vertex is None:
                # print ("hello")
                return find_points(x_0, (x_1+x_0)/2, x_lst, best_vertex)
            # print ("found here")
            return v_0, best_vertex

        else:
            
            if np.any(np.all(x_1 == x_lst, axis=1)): # only happens if we were finding midpoint
                # print ("not in list")
                return x_0, x_1
            if len(x_lst) <= 3:
                return x_0, best_vertex
            best_vertex = x_1
            assert best_vertex in x_lst
            
            # edge case where x too big
            x_1_tmp = x_lst[np.where(x_lst==x_1)[0][0]+1] # next element in list
            
            #x_lst = np.delete(x_lst, np.where(x_lst==x_1)[0])
            x_1 = x_1_tmp
            return find_points(x_0, x_1, x_lst, best_vertex)
    
    #sort the list in ascending x 
    points = points[points[:, 0].argsort()]
    ## print (points)
    v_0 = points[0]
    v_1 = points[1]
    # find the two closest vertices to v_0
    distances = np.sqrt(np.sum((points-v_0)**2,axis=1))
    # # print (distances)
    # dist_idx = np.argpartition(distances, 3)
    # # print (dist_idx)
    #adj_v1, adj_v2 = points[dist_idx[1]], points[dist_idx[2]]
    best_vertex = None
    # return adj_v1, adj_v2
    # print ("",v_0, v_1)
    return find_points(v_0, v_1, points, best_vertex)
