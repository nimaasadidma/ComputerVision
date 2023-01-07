import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace


def get_tiny_images(image_paths):

    feats = list()

    for i in image_paths:
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize (img, (16, 16))
        avr = np.average (img, axis = None, weights = None, returned = False)
        dev = np.var (img, dtype = None, out = None, ddof = 0)
        img = (img - avr) / (dev)
        #img = (img - np.average(img)) / np.std(img)

        img = img.flatten (order = 'A')
        feats.append (img)

    feats = np.asarray (feats)
    return feats




def build_vocabulary(image_paths, vocab_size):


  dim = 128      # length of the SIFT descriptors that you are going to compute.
  vocab = np.zeros((vocab_size,dim))

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  A = len(image_paths)
  B = 220*A
  sifts = np.ones((B, 128))

  ind = 0
    
  for R in range(0, A):
      pic = image_paths[R]
      img = load_image_gray(pic)
      frames, descriptors = vlfeat.sift.dsift(img, fast = True, step = 15) 
      S = descriptors.shape[0]
      index = np.random.permutation(S)
      sifts [ind:ind+features, :] = sifts [index[:features], :]
      ind = ind + features
    
  cluster_centers = vlfeat.kmeans.kmeans(sifts, vocab_size)
  vocab = cluster_centers
  vocab = np.ones((vocab,128))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return vocab






def get_bags_of_sifts(image_paths, vocab_filename):

  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)

  # dummy features variable
  feats = []

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  vocab_size = vocab.shape[0]
  L = len(image_paths)

  for a in range(0, L):
    img = load_image_gray(image_paths[a]).astype('float32')
    position, descriptors= vlfeat.sift.dsift(img,fast = True,step = 10)
    descriptors = descriptors.astype('float32')
    
    Z = np.ones(vocab_size)
    vocab = vocab.reshape(-1, 1)
    descriptors = descriptors.reshape(-1, 1)
    Dist = sklearn_pairwise.pairwise_distances(descriptors, vocab, metric = 'euclidean', n_jobs = None)

    for d in Dist:
      near = np.argmin(d, axis = None, out = None)
      Z[near] = Z[near] + 1

    Z = Z / np.linalg.norm(Z)

    feats.append(Z)

  feats = np.array(feats)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return feats


def category(labels):
 mine = dict()

 for Key, Value in enumerate(list(set(labels))):
   mine[Value] = Key
 return mine

def get_keys(d, val):
  return [k for k,v in d.items() if v == val]



def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='euclidean'):

  test_labels = list()

  Dist = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats)
  Ds = Dist.shape[0]
    
  labels_map = category(train_labels)
  L = len(train_labels)

  train_label = np.ones((1, L), dtype = np.int32)
  
  for a in range(0, L):
    train_label[0, a] = labels_map[train_labels[a]]
    
  index = np.argsort(Dist, axis = 1)
  cluster_labels = np.ones((Dist.shape[0], 1), dtype = np.int32)


  for b in range(0, Ds):
    for c in range(0, 1):
        cluster_labels[b, c] = train_label[0, index[b, c]]
 
  N = cluster_labels.shape[0] 
  for d in range(0, N):

    KL = np.bincount(cluster_labels[d, :])
    KL = np.argmax(KL, axis = None, out = None)
    test_labels.append(get_keys(labels_map, KL)[0])

  return test_labels





def svm_classify(train_image_feats, train_labels, test_image_feats):


    categories = list(set(train_labels))


    svms = {cat: LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5) for cat in categories}


  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

    test_labels = list()

    SUP = LinearSVC(C=700.0, class_weight = None, dual = True, fit_intercept = True,
                    intercept_scaling = 1, loss = 'squared_hinge', max_iter = 1500,
                    multi_class = 'ovr', penalty = 'l2', random_state = 0, tol = 1e-3,
                    verbose = 0)
    SUP.fit(train_image_feats, train_labels)
    
    test_labels = SUP.predict(test_image_feats)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    return test_labels
