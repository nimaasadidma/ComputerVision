import numpy as np

def my_imfilter(image, filter):
 import numpy as np

def my_imfilter(image, filter):

 
    X = image[:,:,0]
    Y = image[:,:,1]
    Z = image[:,:,2]
    
    dimensions = len (filter)
    pad_value = int (dimensions * 0.5)
    padding = [pad_value, pad_value]
    thepad = (padding, padding)
    
    picpad = list()
    picpad.append(np.pad(X, thepad, mode = 'symmetric'))
    picpad.append(np.pad(Y, thepad, mode = 'symmetric'))
    picpad.append(np.pad(Z, thepad, mode = 'symmetric'))
    
    
    output = np.ones_like(Z, order = 'K')
    
    
    X_form = X.shape[0]
    Y_form = Y.shape[1]
    
    
    for i in picpad:
        imgpad = list()
 
        for i1 in range (X_form):
        
            for i2 in range (Y_form):
                
                AllDone = np.sum(np.multiply(i[i1:i1 + dimensions, i2:i2 + dimensions], filter))
                imgpad.append(AllDone)
                
        imgpad = np.array(imgpad)
        imgpad = imgpad.reshape(X_form, Y_form, order = 'C')
        output = np.dstack((output, imgpad))
        
    
    outcome = output[:, :, 1:]

    return outcome


def create_hybrid_image(image1, image2, filter):

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
        
        
  low_frequencies = my_imfilter(image1, filter)
  high_frequencies =  image2 - my_imfilter(image2, filter)
    
  hybrid_img = high_frequencies + low_frequencies
  hybrid_image = np.minimum(1.0, np.maximum(hybrid_img,0.0))

  return low_frequencies, high_frequencies, hybrid_image
