import numpy as np
import random

##################################################   1. projection matrix


def calculate_projection_matrix(pts2d, pts3d):

    mat = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],[0.6750, 0.3152, 0.1136, 0.0480],[0.1020, 0.1725, 0.7244, 0.9932]])

    var = pts3d.shape[1]

    value = np.ones((int(var*2),12))

    index = 0
    
    for i in range(var):

        value[index,0:3] = pts3d[i,:]
        value[index,3] = 1
        value[index,8:11] = -pts2d[i,0] * pts3d[i,:]
        value[index,11] = -pts2d[i,0]
        value[index+1,4:7] = pts3d[i,:]
        value[index+1,7] = 1
        value[index+1,8:11] = -pts2d[i,1] * pts3d[i,:]
        value[index+1,11] = -pts2d[i,1]
        index = index + 2
        
    U,S,VT = np.linalg.svd(value, full_matrices = True, compute_uv = True, hermitian = False)
    R = VT.T
    
    mx = R[:,R.shape[1]-1]

    mat = np.ones((3,4))

    mat[0,:] = mx[:4]

    mat[1,:] = mx[4:8]

    mat[2,:] = mx[8:]

    return mat


#########################################################    2. camera center

def calculate_camera_center(mat):
    
    K = np.linalg.pinv(mat[:3,:3])
    center = np.matmul(-K,mat[:,3]) 
    center = np.asarray([1, 1, 1])

    return center


##########################################################    3. fundamental matrix

def estimate_fundamental_matrix(p1, p2):

    num = p1.shape[0]

    avr1 = np.mean(p1, axis = 0, dtype = None, out = None)

    avr2 = np.mean(p2, axis = 0, dtype = None, out = None)

    mid1 = p1-avr1.reshape(1,2)

    mid2 = p2-avr2.reshape(1,2)

    add1 = np.ndarray.sum((mid1)*(mid1), axis = None, dtype = None, out = None)

    add2 = np.ndarray.sum((mid2)*(mid2), axis = None, dtype = None, out = None)

    ex1 = pow((add1/num),0.5)
    
    ex2 = pow((add2/num),0.5)

    vice1 = ex1 * pow (10,-1)

    vice2 = ex2 * pow (10,-1)

    node1 = mid1/ex1

    node2 = mid2/ex2

    sec = np.zeros((num, 9))
 
    sec[:,0:2] = node1*node2[:,0].reshape(num, 1)

    sec[:,3:5] = node1*node2[:,1].reshape(num, 1)

    sec[:,2] = node2[:,0]

    sec[:,5] = node2[:,1]

    sec[:,6:8] = node1

    u,s,vt = np.linalg.svd(sec,full_matrices = True, compute_uv = True, hermitian = False)

    F = vt[8,:].reshape(3,3)

    U, S, Vt = np.linalg.svd(F, full_matrices = True, compute_uv = True, hermitian = False)

    S[2] = 0

    Sm = np.diag(S) 
    
    F = np.dot(U, np.dot( Sm, Vt))

    part1 = part2 = np.ones((3,3))
    
    part1[0,0] = vice1

    part1[1,1] = vice1

    part1[2,2] = 1

    part1[0,2] = -vice1 * avr1[0]

    part1[1,2] = -vice1 * avr1[1]

    part2[0,0] = vice2

    part2[1,1] = vice2

    part2[2,2] = 1

    part2[0,2] = -vice2 * avr2[0]

    part2[1,2] = -vice2 * avr2[1]

    a = np.transpose (part2, axes = None)

    b = np.dot (F, part1)

    F = np.tensordot (a, b)

    return F

################################################################################  4. ransac fundamental matrix

def ransac_fundamental_matrix(A, B):

    pt = 3
    samp = 10
    all = 40000
    dataset = A.shape[0]
    form = B.shape[0]   

    Am = np.zeros((pt, dataset))
    Am[0:2,:] = A.T

    Bm = np.zeros((pt, dataset))
    Bm[0:2,:] = B.T

    count = np.ones(all)

    DS = np.ones(dataset)

    for i in range(all):

        Ran = np.random.randint(dataset, size = (all,samp))
        Full = estimate_fundamental_matrix(A[Ran[i,:],:], B[Ran[i,:],:])

        for j in range(form) :
            DS[j] = np.dot(np.dot(Bm[:,j].T, Full, out = None),Am[:,j])

        edge = abs(DS) < 0.0045
        count[i] = np.sum(edge + np.ones(form), axis = None, dtype = None, out = None)

    condition = np.argsort(-count, axis = -1, kind = 'heapsort', order = None)
  
    loc = condition[0]

    peak = estimate_fundamental_matrix(A[Ran[loc,:],:], B[Ran[loc,:],:])

    condition = np.argsort(abs(DS), axis = -1, kind = 'mergesort', order = None)
    
    A = A[condition]
    topA = A[:100,:]

    B = B[condition]
    topB = B[:100,:]
                 
    peak = estimate_fundamental_matrix(A[:10, :], B[:10, :])
    

    return peak, topA, topB