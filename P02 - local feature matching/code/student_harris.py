import cv2
import numpy as np
import math

def ANMS (x , y, r, maximum):

    index = 0
    count = 0
    mylist = list()

    while index < len(x):

        num = 20000000000

        x1_Coordinate = x[index]
      
        y1_Coordinate = y[index]

        

        while count < len(x):
            
            x2_Coordinate = x[count]
            y2_Coordinate = y[count]
            
            x_dif = x2_Coordinate - x1_Coordinate
            y_dif = y2_Coordinate - y1_Coordinate

            if (x1_Coordinate != x2_Coordinate and y1_Coordinate != y2_Coordinate) and r[index] < r[count]:

                distance =((x_dif)*(x_dif) + (y_dif)*(y_dif))**0.5

                if distance < num:

                    num = distance

            count += 1 

        mylist.append([x1_Coordinate, y1_Coordinate, num])
        
        index += 1
        count = 0

    mylist.sort(key = lambda t: t[2])
    mylist = mylist[len(mylist)-maximum:len(mylist)]

    return mylist




def get_interest_points(image, feature_width):

    img_width = image.shape[0]
    img_height = image.shape[1]
       
    xedge = list()
    yedge = list()
    val = list()

    border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
    x_factor = cv2.Sobel(image, cv2.CV_64F,1,0,ksize = 5, borderType = border)
    y_factor = cv2.Sobel(image, cv2.CV_64F,0,1,ksize = 5, borderType = border)

    Mx  = pow(x_factor,2)
    My  = pow(y_factor,2)
    Mxy = x_factor * y_factor

    for m in range(16, img_width - 16):
        
        for n in range(16, img_height - 16):

            Ixx = Mx[m-1:m+1 , n-1:n+1]
            Iyy = My[m-1:m+1 , n-1:n+1]
            Ixy = Mxy[m-1:m+1, n-1:n+1]

            sumx = Ixx.sum()
            sumy = Iyy.sum()
            sumxy = Ixy.sum()

            Det = sumx*sumy - sumxy*sumxy
            addxy = pow((sumx + sumy),2)
            sec = addxy * 0.04
            res = Det - sec

            if res > 9999:

                xedge.append(n)
                yedge.append(m)
                val.append(res)


    xedge = np.asarray(xedge)
    yedge = np.asarray(yedge)
    val = np.asarray(val)

    angle = ANMS(xedge, yedge, val, 3025)

    angle = np.asarray(angle)

    x = angle[:,0]
    y = angle[:,1]
    scales = angle[:,2]


    return x, y, scales