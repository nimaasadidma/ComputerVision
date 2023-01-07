import numpy as np
import math


def match_features(f1, f2, x1, y1, x2, y2):
        
    Val = list()
    hx = list()
    hy = list()
    
    a = f1.shape[0]
    b = f2.shape[0]
    Distance = np.ones((a, b))

    
    for i in range(a):

        for j in range(b):

            fimg1 = f1[[i],:]
            fimg2 = f2[[j],:]


            dif = pow ((fimg1 - fimg2), 2)

            add = dif.sum()

            add = pow(add, 0.5)

            Distance[i,j] = add


        position = np.argsort(Distance[i,:],kind = 'heapsort',  axis = -1)
       
        vec1 = position[0]
        vec2 = position[1]
        pos1 = Distance[i,vec1]
        pos2 = Distance[i,vec2]
        edge = pos1/pos2

        if edge <0.80 :

            hx.append(i)

            hy.append(vec1)

            Val.append(pos1)

    confidences = np.asanyarray(Val)
    
    pos_X = np.asanyarray(hx)
    pos_Y = np.asanyarray(hy)
    similarities = np.stack((pos_X, pos_Y), axis = -1)
    



    return similarities, confidences