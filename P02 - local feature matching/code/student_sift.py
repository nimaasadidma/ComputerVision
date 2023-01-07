import numpy as np
import cv2
import math


def get_features(image, x, y, feature_width):

  x =  (np.trunc(x)).astype(np.int32)
  y =  (np.trunc(y)).astype(np.int32)
  xdir = len (x)
  ydir = len (y)

  Filter = cv2.getGaussianKernel(4, 10)
  Filter = np.dot(Filter, Filter.T)

  image = cv2.filter2D(image, -1, Filter)

  Normalise = np.ones((xdir,128))


  for i1 in range(xdir):

    xpattern = int(x[i1])
    ypattern = int(y[i1])
    
    box = image[ypattern-8:ypattern + 8, xpattern-8:xpattern + 8] 

    for i2 in range(4): 
      for i3 in range(4):

        frame = box[i2*4:i2*4 +4,i3*4: i3*4+4]
        cut = cv2.copyMakeBorder(frame, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        shape = np.ones((4,4))
        attitude = np.ones((4,4))
        
        S1 = frame.shape[0]
        S2 = frame.shape[1]

        for i4 in range(S1):
          for i5 in range(S2):

            shape[i4,i5] = math.sqrt((cut[i4+1,i5] - cut[i4-1,i5])**2 + (cut[i4,i5+1] - cut[i4,i5-1])**2)
            attitude[i4,i5] = np.arctan2((cut[i4+1,i5] - cut[i4-1,i5]),(cut[i4,i5+1] - cut[i4,i5-1]))

        shape = shape
        condition = attitude * 30
        fig, border = np.histogram(condition, bins = 8, range = (-180, 180), weights = shape, density = False)
        
        for i6 in range(8):
          n = i6 + i3*8 + i2*32 
          Normalise[i1,n] = fig[i6]


  R1 = Normalise.shape[0]
  R2 = Normalise.shape[1]
  
  for i7 in range(R1):
    plus = 0
    
    for i8 in range(R2): 
      plus = plus + pow((Normalise[i7][i8]),2)
    plus = math.sqrt(plus)

    for i9 in range(R2):
      Normalise[i7][i9] = Normalise[i7][i9]/plus

  finalise = Normalise

  return finalise