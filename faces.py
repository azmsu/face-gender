'''CSC411: Assignment 1
Due: Friday, February 03, 2017
Author: Zi Mo Su (1001575048)
'''


#------------------------------------IMPORTS------------------------------------
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


#-----------------------------------FUNCTIONS-----------------------------------
# PART 1 -----------------------------------------------------------------------
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/
    '''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def crop(img, coords):
    '''Return cropped RGB image, represented as a numpy array of size n x m x 3.
    Image is cropped according to coordinates given in data file.
    Arguments:
    img -- raw uncropped image file.
    coords -- coordinates of the crop in the form [x1, y1, x2, y2], where
    (x1, y1) is the top-left pixel and (x2, y2) is the bottom right pixel.
    '''
    # create image array
    img_array = imread(img, mode="RGB")
    return img_array[int(coords[1]):int(coords[3]), int(coords[0]):int(coords[2])]
    
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1.
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255.

def create_set(act):
    '''Return set containing all pictures of each actor/actress in the list act
    in the form of a dictionary with key-value pairs {(name###):(2D array)}.
    Arguments:
    act -- list of actors/actresses
    '''
    testfile = urllib.URLopener()
    set = {}
    
    # loop through each actor/actress in act
    for a in act:
        name = a.split()[1].lower()
        i = 0
        
        # loop through each line in the raw data (every image of every 
        # actor/actress is in 'faces_subset.txt'
        for line in open("faces_subset.txt"):
            if a in line:
                # filename is of the form: '[name][###].[ext]'
                filename = name+str(i).zfill(3)+'.'+line.split()[4].split('.')[-1]
                # A version without timeout (uncomment in case you need to 
                # unsupress exceptions, which timeout() does)
                # testfile.retrieve(line.split()[4], "uncropped/"+filename)
                # timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                
                # remove images that are unreadable
                try:
                    imread("uncropped/"+filename)
                except IOError:
                    if os.path.isfile("uncropped/"+filename):
                        os.remove("uncropped/"+filename)
                    print "IOError"
                    continue
                
                if not os.path.isfile("uncropped/"+filename):
                    continue
                
                print filename
                
                # crop image
                coords = line.split("\t")[4].split(",")
                img = crop("uncropped/"+filename, coords)
                # imsave("cropped/crop_"+filename, img)
                
                # convert image to grayscale
                img = rgb2gray(img)
                # imsave("gray/gray_"+filename, img)
                
                # resize image to 32 x 32
                try:
                    img = imresize(img, (32,32))
                except ValueError:
                    os.remove("uncropped/"+filename)
                    print "ValueError"
                    continue
                    
                # imsave("resized/rs_"+filename, img)
                
                # store image in set
                set[name+str(i).zfill(3)] = img
                
                i += 1
     
    return set

# PART 2 -----------------------------------------------------------------------
def partition(act, set, train_size = 100, val_size = 10, test_size = 10):
    '''Returns three subsets from the set of all actors/actresses; training,
    validation, and test sets. Each subset's size can be specified and the list
    of actors/actresses to retain in the subset is specified.
    Arguments:
    act -- list of actors/actresses to retain in subsets
    set -- input set of all actors/actresses
    train_size -- number of images per actor/actress to retain in training set
    val_size -- number of images per actor/actress to retain in validation set
    test_size -- number of images per actor/actress to retain in test set
    '''
    val_set = {}
    test_set = {}
    train_set = {}
    
    # loop through actors/actresses to be retained
    for a in act:
        name = a.split()[1].lower()
        
        # loop through each element in the dictionary set
        for filename in set:
            if name in filename:
                
                # save first val_size number of actors to validation set
                if int(filename[-3:]) < val_size:
                    val_set[filename] = set[filename]
                    
                # save next test_size number of actors to test set
                elif int(filename[-3:]) < val_size + test_size:
                    test_set[filename] = set[filename]
                    
                # save next train_size number of actors to training set
                elif int(filename[-3:]) < val_size + test_size + train_size:
                    train_set[filename] = set[filename]
                    
                else:
                    continue
    
    return train_set, val_set, test_set

# PART 3 -----------------------------------------------------------------------
def f(X, y, t, m):
    '''Returns the result of the cost function.
    Arguments:
    X -- matrix of size m x 1025 containing image data
    y -- array of size 1 x m containing -1s and 1s depending on classification
    t -- array of size 1 x 1025 containing theta parameters
    m -- the number of images
    '''
    return 1/(2.*m)*sum((y.T - dot(X, t.T))**2)

def df(X, y, t, m):
    '''Returns the gradient of the cost function.
    Arguments:
    X -- matrix of size m x 1025 containing image data
    y -- array of size 1 x m containing -1s and 1s depending on classification
    t -- array of size 1 x 1025 containing theta parameters
    m -- the number of images
    '''
    return -1./m*sum((y.T - dot(X, t.T))*X, 0)
    
def make_row(arr):
    '''Returns a 1 x 1025 array representing the image. The first element is 1
    and the next 1024 are the pixels of the image read from left to right and
    top to bottom.
    Arguments:
    arr -- 2D numpy array, of size 32 x 32, that represents the image
    '''
    # reshape x into (1 x 1024) and insert 1 at beginning
    x = np.reshape(arr, 1024)
    x = np.hstack((array([1]), x))
    return x
    
def grad_descent(X, y, init_t, alpha, m, vector = False):
    '''Returns the theta parameters for which the cost function is minimized,
    through numerical computation using gradient descent.
    Arguments:
    X -- matrix containing image data
    y -- array containing -1s and 1s or 1 x 6 arrays if vectorized, depending on 
         classification
    init_t -- array containing initial theta parameters
    alpha -- parameter used for gradient descent, 'learning rate'
    m -- the number of images
    vector -- set to True to enable vectorized gradient descent
    '''
    EPS = 1e-6
    prev_t = init_t - 10*EPS
    t = init_t.copy()
    max_iter = 100000
    iter = 0
    
    while norm(t-prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        if vector == True:
            t -= alpha*df_v(X, y, t, m)
        else:
            t -= alpha*df(X, y, t, m)
        iter += 1
        
    if vector == True:
        cost = f_v(X, y, t, m)
    else:
        cost = f(X, y, t, m)
        
    print "Minimum found at", t, "with cost function value of", cost, "on iteration", iter
    return t

def train_hc(train_set, a, m, run_gd = True):
    '''Returns the theta vector for which the cost function is minimized, if
    run_gd is True, otherwise the constructed 2D array X and the array y are
    returned.
    Arguments:
    train_set -- training set for hader/carell classification, if run_gd is
                 False, this set is just the set used to create X and y
    a -- value of alpha for gradient descent
    m -- total number of images
    run_gd -- set to False if gradient descent is not desired
    '''
    X = []
    y = array([])
    
    for filename in train_set:
        # hader classified as -1
        if "hader" in filename:
            x = make_row(train_set[filename])
            y_temp = -1
        # carell classified as 1
        elif "carell" in filename:
            x = make_row(train_set[filename])
            y_temp = 1
        else:
            continue
        
        if X == []:
            X = array([x])
        else:    
            X = np.append(X, [x], 0)
        
        y = np.append(y, y_temp)
    
    # initial theta
    t0 = array([np.zeros(1025)])
    
    y = array([y])
    
    if run_gd == True:
        return grad_descent(X, y, t0, a, m)
    else:
        return X, y

def performance_hc(set, t):
    '''Evaluates the performance of a given set for hader/carell classification.
    Arguments:
    set -- set to be evaluated
    t -- theta vector generated through training
    '''
    right = 0
    wrong = 0
    
    for filename in set:
        if "hader" in filename:
            x = make_row(set[filename])
            # x*t < 0 -> -1 (hader)
            if dot(x, t.T) < 0:
                right += 1
            else:
                wrong += 1
    
        if "carell" in filename:
            x = make_row(set[filename])
             # x*t > 0 -> 1 (carell)
            if dot(x, t.T) >= 0:
                right += 1
            else:
                wrong += 1
                
    if set == train_set:
        set = "Training set"
    elif set == val_set:
        set = "Validation set"
    else:
        set = "Testing set"
        
    print set, "is", str(right/((right + wrong)*0.01)) + "% accurate"

# PART 4 -----------------------------------------------------------------------
def get_rn(min, max, n):
    '''Returns array of n random numbers (no repeats) between min and max 
    (inclusive).
    Arguments:
    min -- minimum number that can be chosen
    max -- maximum number that can be chosen
    n -- number of numbers chosen
    '''
    arr = np.arange(min, max+1)
    np.random.shuffle(arr)
    return arr[:n]

# PART 5 -----------------------------------------------------------------------
def train_gender(train_set, a, n):
    '''Returns the theta vector for which the cost function is minimized by
    using the training set to train the gender classifier. The number of
    training examples used is specified by n per actor/actress for a total of 
    6n.
    Arguments:
    train_set -- training set
    a -- alpha value used for gradient descent
    n -- number of each actor/actress, total size of 6n
    '''
    
    X = []
    Y = array([])
    
    # males are classified as -1 and females as 1
    for filename in train_set:
        if "baldwin" in filename and int(filename[-3:])-20 < n:
            x = make_row(train_set[filename])
            y = -1
        elif "hader" in filename and int(filename[-3:])-20 < n:
            x = make_row(train_set[filename])
            y = -1
        elif "carell" in filename and int(filename[-3:])-20 < n:
            x = make_row(train_set[filename])
            y = -1
        elif "drescher" in filename and int(filename[-3:])-20 < n:
            x = make_row(train_set[filename])
            y = 1
        elif "ferrera" in filename and int(filename[-3:])-20 < n:
            x = make_row(train_set[filename])
            y = 1
        elif "chenoweth" in filename and int(filename[-3:])-20 < n:
            x = make_row(train_set[filename])
            y = 1
        else:
            continue
            
        if X == []:
            X = array([x])
        else:    
            X = np.append(X, [x], 0)
            
        Y = np.append(Y, y)
     
    # make y correct dimension (1 x 6n)
    Y = array([Y])
    
    # initialize theta (1 x 1025)
    t0 = array([np.zeros(1025)])
    
    # Note: X has dimensions 6n x 1025
    
    return grad_descent(X, Y, t0, a, 6*n)
    
def performance_gender(set, t):
    '''Evaluates the performance of a given set for gender classification.
    Arguments:
    set -- set to be evaluated
    t -- theta vector generated through training
    '''
    right = 0
    wrong = 0
    
    for filename in set:
        # male
        if "baldwin" in filename or "hader" in filename or "carell" in filename or "butler" in filename or "radcliffe" in filename or "vartan" in filename:
            x = make_row(set[filename])
            # x*t < 0 -> -1 (male)
            if dot(x, t.T) < 0:
                right += 1
            else:
                wrong += 1
    
        else:
            x = make_row(set[filename])
            # x*t > 0 -> 1 (female)
            if dot(x, t.T) >= 0:
                right += 1
            else:
                wrong += 1
                
    if set == train_set:
        set = "Training set"
    elif set == val_set:
        set = "Validation set"
    else:
        set = "Testing set"
        
    print set, "is", str(right/((right + wrong)*0.01)) + "% accurate"

# PART 6 -----------------------------------------------------------------------
def f_v(X, Y, THETA, m):
    '''Returns the result of the cost function. The cost function is 
    (THETA.T*X-Y)^2 (element wise square) summed vertically and then 
    horizontally.
    Arguments:
    X -- matrix of size n x m containing image data
    Y -- array of size k x m containing classification data
    THETA -- array of size n x k containing theta parameters to be minimized 
    where:
    n -- number of pixels per image + 1
    m -- number of training examples
    k -- number of possible labels
    '''
    return (1./(2*m))*sum(sum((dot(THETA.T, X) - Y)**2, 0))
    
def df_v(X, Y, THETA, m):
    '''Returns the vector gradient of the cost function.
    Arguments:
    X -- matrix of size n x m containing image data
    Y -- array of size k x m containing classification data
    THETA -- array of size n x k containing theta parameters to be minimized 
    where:
    n -- number of pixels per image + 1
    m -- number of training examples
    k -- number of possible labels
    '''
    return (1./m)*dot(X, (dot(THETA.T, X) - Y).T)
    
def finite_diff(X, Y, THETA, m, h):
    '''Returns the finite difference calculation of the gradient of the cost 
    function defined by f_v.
    X -- matrix of size n x m containing image data
    Y -- array of size k x m containing classification data
    THETA -- array of size n x k containing theta parameters to be minimized 
    h -- step size
    where:
    n -- number of pixels per image + 1
    m -- number of training examples
    k -- number of possible labels
    '''
    G = np.zeros((1025, 6))
    for i in range(1025):
        for j in range(6):
            H = np.zeros((1025, 6))
            H[i,j] = h
            # compute the partial derivative of f with respect to theta_ij and
            # save it in the gradient G
            G[i,j] = (f_v(X, Y, THETA + H, m) - f_v(X, Y, THETA, m))/(h*1.)
    return G
    
# PART 7 -----------------------------------------------------------------------
def train_fr(train_set, a):
    '''Returns the theta matrix for which the cost function is minimized using 
    the training set to train the face recognition classifier.
    Arguments:
    train_set -- training set
    a -- value of alpha for gradient descent
    '''
    X = []
    Y = []
    
    for filename in train_set:
        if "drescher" in filename:
            x = make_row(train_set[filename])
            y = array([1,0,0,0,0,0])
        elif "ferrera" in filename:
            x = make_row(train_set[filename])
            y = array([0,1,0,0,0,0])
        elif "chenoweth" in filename:
            x = make_row(train_set[filename])
            y = array([0,0,1,0,0,0])
        elif "baldwin" in filename:
            x = make_row(train_set[filename])
            y = array([0,0,0,1,0,0])
        elif "hader" in filename:
            x = make_row(train_set[filename])
            y = array([0,0,0,0,1,0])
        elif "carell" in filename:
            x = make_row(train_set[filename])
            y = array([0,0,0,0,0,1])
        else:
            continue
        
        if X == []:
            X = array([x])
        else:    
            X = np.append(X, [x], 0)
        
        if Y == []:
            Y = array([y])
        else:
            Y = np.append(Y, [y], 0)
            
    X = X.T  # n x m = 1025 x 600
    Y = Y.T  # k x m = 6 x 600
    
    T0 = np.zeros((1025, 6))  # n x k = 1025 x 6
    
    return grad_descent(X, Y, T0, a, 600, True), X, Y, T0

def performance_fr(set, T):
    '''Evaluates the performance of a given set for facial recognition
    classification.
    Arguments:
    set -- set to be evaluated
    T -- theta matrix generated through training
    '''
    right = 0
    wrong = 0
    
    for filename in set:
        # drescher
        if "drescher" in filename:
            x = make_row(set[filename])
            # (T.T*X)[0] is max in T.T*X ->  (drescher)
            if max(dot(T.T, x.T)) == dot(T.T, x.T)[0]:
                right += 1
            else:
                wrong += 1
        # ferrera
        elif "ferrera" in filename:
            x = make_row(set[filename])
            # (T.T*X)[1] is max in T.T*X ->  (ferrera)
            if max(dot(T.T, x.T)) == dot(T.T, x.T)[1]:
                right += 1
            else:
                wrong += 1
        # chenoweth
        elif "chenoweth" in filename:
            x = make_row(set[filename])
            # (T.T*X)[2] is max in T.T*X ->  (chenoweth)
            if max(dot(T.T, x.T)) == dot(T.T, x.T)[2]:
                right += 1
            else:
                wrong += 1
        # baldwin
        elif "baldwin" in filename:
            x = make_row(set[filename])
            # (T.T*X)[3] is max in T.T*X ->  (baldwin)
            if max(dot(T.T, x.T)) == dot(T.T, x.T)[3]:
                right += 1
            else:
                wrong += 1
        # hader
        elif "hader" in filename:
            x = make_row(set[filename])
            # (T.T*X)[4] is max in T.T*X ->  (hader)
            if max(dot(T.T, x.T)) == dot(T.T, x.T)[4]:
                right += 1
            else:
                wrong += 1
        # carell
        elif "carell" in filename:
            x = make_row(set[filename])
            # (T.T*X)[5] is max in T.T*X ->  (carell)
            if max(dot(T.T, x.T)) == dot(T.T, x.T)[5]:
                right += 1
            else:
                wrong += 1
        else:
            continue
    
    if set == train_set:
        set = "Training set"
    elif set == val_set:
        set = "Validation set"
    else:
        set = "Testing set"
        
    print set, "is", str(right/((right + wrong)*0.01)) + "% accurate"


#-----------------------------------MAIN CODE-----------------------------------
act = ["Fran Drescher", "America Ferrera", "Kristin Chenoweth", "Alec Baldwin", "Bill Hader", "Steve Carell"]
act_test = ["Gerard Butler", "Daniel Radcliffe", "Michael Vartan", "Lorraine Bracco", "Peri Gilpin", "Angie Harmon"]

# PART 1 -----------------------------------------------------------------------
print "Running Part 1..."
if not os.path.exists("act_set.npy"):
    act_set = create_set(act)
    np.save("act_set.npy", act_set)
    
else:
    act_set = np.load("act_set.npy").item()
print "Part 1 Complete!\n"

# PART 2 -----------------------------------------------------------------------
print "Running Part 2..."
train_set, val_set, test_set = partition(act, act_set)
print "Part 2 Complete!\n"

# PART 3 -----------------------------------------------------------------------
print "Running Part 3..."
t = train_hc(train_set, a=1e-7, m=200)
X, y = train_hc(train_set, a=1e-7, m=20, run_gd=False)

print "Cost for the validation set is:", f(X, y, t, m=20)

performance_hc(train_set, t)
performance_hc(val_set, t)
performance_hc(test_set, t)
print "Part 3 Complete!\n"

# PART 4 -----------------------------------------------------------------------
print "Running Part 4..."
# training using the full training set
# first remove theta_0 at front of t
t = t[:,1:]
t = np.reshape(t, (32, 32))
imsave("part4_1.jpg", t)

# training using 2 images of each actor, randomly selected
rn = get_rn(20, 119, 4)
train_set = {"hader"+str(rn[0]).zfill(3):train_set["hader"+str(rn[0]).zfill(3)],
             "hader"+str(rn[1]).zfill(3):train_set["hader"+str(rn[1]).zfill(3)],
             "carell"+str(rn[2]).zfill(3):train_set["carell"+str(rn[2]).zfill(3)],
             "carell"+str(rn[3]).zfill(3):train_set["carell"+str(rn[3]).zfill(3)],}

t = train_hc(train_set, a=1e-8, m=4)
t = t[:,1:]
t = np.reshape(t, (32, 32))
imsave("part4_2.jpg", t)
print "Part 4 Complete!\n"

# PART 5 -----------------------------------------------------------------------
print "Running Part 5..."
for i in range(10):
    train_set, val_set, test_set = partition(act, act_set, 10*(i+1))
    t = train_gender(train_set, a=1e-7, n=10*(i+1))
    print "For training set of size", str(60*(i+1)) +":"
    performance_gender(train_set, t)
    performance_gender(val_set, t)
    performance_gender(test_set, t)
    print

# test on set of other actors
if not os.path.exists("act_test_set.npy"):
    act_test_set = create_set(act_test)
    np.save("act_test_set.npy", act_test_set)

else:
    act_test_set = np.load("act_test_set.npy").item()    

train_set, val_set, test_set = partition(act, act_set, 100)
t = train_gender(train_set, a=1e-7, n=100)
print "Testing on the set of other actors:"
performance_gender(act_test_set, t)
print "Part 5 Complete!\n"

# PART 6 -----------------------------------------------------------------------
print "Running Part 6..."
train_set, val_set, test_set = partition(act, act_set)
T, X, Y, T0 = train_fr(train_set, a=1e-7)
for i in range(5):
    X_test = X
    Y_test = Y
    T_test = T0
    grad_analytic = df_v(X_test, Y_test, T_test, m=600)
    grad_fd = finite_diff(X_test, Y_test, T_test, m=600, h=10**-(5+i))
    print "For h =", str(10**-(5+i)) + ":"
    print "The norm of the difference between the analytical and finite difference gradient is:", norm(grad_fd-grad_analytic)
    print
print "Part 6 Complete!\n"

# PART 7 -----------------------------------------------------------------------
print "Running Part 7..."
performance_fr(train_set, T)
performance_fr(val_set, T)
print "Part 7 Complete!\n"

# PART 8 -----------------------------------------------------------------------
print "Running Part 8..."
for i in range(6):
    t = T[1:,i]
    t = np.reshape(t, (32, 32))
    imsave("part8_"+str(i+1)+".jpg", t)
print "Part 8 Complete!\n"
