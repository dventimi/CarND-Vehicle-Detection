
# Once activated this environment is used to launch Python in whatever
#   way one likes, such as a [[https://www.python.org/shell/][Python shell]], a [[https://ipython.org/][IPython shell]], or a [[http://jupyter.org/][jupyter
#   notebook]].  Having done that, the usual first step is to import the
#   packages that are used.

from glob import glob
from itertools import groupby, islice, zip_longest, cycle, filterfalse, chain
from moviepy.editor import VideoFileClip, VideoClip
from random import choice, sample
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

# Histogram of Oriented Gradients (HOG)

#   The first step in this project was to create a vehicle classifier
#   that was capable of identifying vehicles in an image, since we can
#   treat the frames of the video we are processing as individual
#   images.  Broadly, there are two main approaches for this task.  

#   - [[https://en.wikipedia.org/wiki/Computer_vision][Computer Vision]]
#   - [[https://en.wikipedia.org/wiki/Convolutional_neural_network][Neural Networks]]

#   Because the emphasis of this module in the Udacity course seemed to
#   focus on the computer vision approach, and because we already used
#   neural networks for two previous projects, I chose to explore the
#   first of these.

#   Though the two are similar (and have the same objective:
#   classification), one of the hallmarks of the computer vision
#   approach seems to be manual [[https://en.wikipedia.org/wiki/Feature_extraction][feature extraction]]: relying on human
#   experience to select useful features for a [[https://en.wikipedia.org/wiki/Machine_learning][machine learning]] problem.
#   In the class, we explored several.

#   - [[https://en.wikipedia.org/wiki/Color_space][color space]] selection
#   - [[https://en.wikipedia.org/wiki/Data_binning][binning]] of spatial features
#   - [[https://en.wikipedia.org/wiki/Color_histogram][color histograms]]
#   - [[http://www.learnopencv.com/histogram-of-oriented-gradients/][histogram of oriented gradients (HOG)]]
#   - hybrid approaches

#   I experimented with each of these approaches, and with various
#   combinations of them, and finally selected a simple combination:
#   color space transformation and HOG-features.  Specifically, after
#   trying several different color spaces, I settled on the [[https://en.wikipedia.org/wiki/HSL_and_HSV][HSV]] color
#   space and then performed HOG feature extraction on just the /V/
#   ("value") channel.  Such a simple feature extractor may seem overly
#   simple---and perhaps it is---but the "proof is in the pudding," as
#   they say.  It performed well, with decent accuracy on a test sample
#   (~98%) and on the project video.  Moreover, it has the virtue of
#   requiring relatively few computational resources.  Anything that
#   increases performance is a big win, since it promotes rapid,
#   [[https://en.wikipedia.org/wiki/Iterative_and_incremental_development][iterative]] experimentation.  

#   Moreover, for the HOG parameters (orientations, pixels_per_cell,
#   cells_per_block), I started with the values that we used in the
#   quizzes in the lectures, and then manually tuned toward better
#   values by simple trial-and-error.  Because the classifier seemed to
#   do well on the test set, not much tuning was necessary.

#   Let's dive into some code to see how that went.

#   To set the stage, we were provided with two data archive files,
#   [[file:vehicles.zip][vehicles.zip]] and [[file:non-vehicles.zip][non-vehicles.zip]], which as the names suggest
#   contained images of vehicles and things that are not vehicles.

img1 = mpimg.imread("vehicles/GTI_MiddleClose/image0000.png")
img2 = mpimg.imread("non-vehicles/GTI/image1.png")
fig = plt.figure()
plt.subplot(121)
plt.imshow(img1)
plt.title('Vehicle')
plt.subplot(122)
plt.imshow(img2)
plt.title('Non-Vehicle')
fig.tight_layout()
plt.savefig("output_images/car-examples.png")

# #+RESULTS:
#   : 
#   : >>> >>> <matplotlib.axes._subplots.AxesSubplot object at 0x7f8fb8834b70>
#   : <matplotlib.image.AxesImage object at 0x7f8fb8898438>
#   : <matplotlib.text.Text object at 0x7f8fb87f1780>
#   : <matplotlib.axes._subplots.AxesSubplot object at 0x7f8fb8898828>
#   : <matplotlib.image.AxesImage object at 0x7f8fb87d72b0>
#   : <matplotlib.text.Text object at 0x7f8fb8816eb8>

#   Here is a side-by-side comparison of a vehicle image and a
#   non-vehicle image, drawn from our training sample.

#   [[file:output_images/car-examples.png]]

#   The size of each image is 64 x 64 pixels, and the vehicle and
#   non-vehicle images are contained (after unpacking the archive files)
#   in directories ~vehicle~ and ~non-vehicle~, respectively.  Now,
#   whatever classifier we use, we have to start by reading in these
#   images one way or another.  Confronted with tasks like this, I like
#   to compose small functions based on Python [[http://davidaventimiglia.com/python_generators.html][generators]], so first I
#   define a handful of useful utility functions.  

#   - feed :: generator function over a [[https://docs.python.org/2/library/glob.html][glob]], which maps a value ~y~ to
# 	    each filename that matches ~pattern~, yielding [[https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences][tuples]]
#   - shuffle :: list-builder function over a sequence of tuples, which
# 	       [[https://www.merriam-webster.com/dictionary/reify][reifies]] it into a randomized list
#   - scale :: non-generator function, which scales the values in an
# 	     array, either by the maximum value in the array or by a
# 	     supplied parameter ~maxval~
#   - load :: generator function over a sequence of [[https://en.wikipedia.org/wiki/Ordered_pair][ordered pairs]] in
# 	    which the first element is an image filename and the
# 	    second is any value (perhaps provided by the ~feed~
# 	    function above), which loads the image files into NumPy
# 	    arrays
#   - flip :: generator function over a sequence of ordered pairs in
# 	    which the first element is a NumPy array and the second is
# 	    any value, which "flips" the array horizontally (i.e.,
# 	    across a vertical axis) and producing a mirror image
#   - mirror :: generator function over a sequence of ordered pairs as
# 	      in ~flip~, but which yields first the entire sequence
# 	      unchanged and then the entire sequence again but with the
# 	      images flipped

#   These are implemented as one-liners.

feed = lambda pattern, y: ((f, y) for f in glob(pattern))
shuffle = lambda l: sample(l, len(l))
scale = lambda img,maxval=None: (img/np.max(img)*255).astype(np.uint8) if maxval==None else (img/maxval*255).astype(np.uint8)
load = lambda g: ((mpimg.imread(x[0]),x[1]) for x in g)
flip = lambda g: ((x[0][:,::-1,:],x[1]) for x in g)
mirror = lambda g: chain(g, flip(g))

# #+RESULTS:

#   When composed together, these functions provide a generator that
#   [[https://en.wikipedia.org/wiki/Lazy_loading][lazily loads]] training images in random order, twice: first
#   unflipped, and second flipped.  This serves several related
#   purposes.  First, randomizing data for training purposes is a
#   best-practice in Machine Learning.  Second, it effectively doubles
#   the size of our training set.  Third, we anticipate encountering
#   vehicles on the road from any angle, where the vehicles themselves
#   are inherently symmetric across a vertical plane running
#   longitudinally down the length of the car.

#   Before we can use this generator, however, we need something to use
#   it on.  Let's define our functions for extracting features and for
#   creating our classifiers.

#   First, the ~extract_features~ function transforms a given image to a
#   target color space, performs HOG feature extraction on a target
#   color channel, then [[http://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range][scales]] the features.

def extract_features(img,
                     colorspace=cv2.COLOR_RGB2HSV,
                     channel=2,
                     orient=9,
                     pix_per_cell=8,
                     cell_per_block=4,
                     transform_sqrt=False,
                     feature_vec=True):
    img = scale(img)
    X = np.array([])
    X = np.append(X,
                  hog(cv2.cvtColor(img, colorspace)[:,:,channel],
                      orient,
                      (pix_per_cell,pix_per_cell),
                      (cell_per_block,cell_per_block),
                      transform_sqrt = transform_sqrt,
                      feature_vector = feature_vec))
    s = StandardScaler().fit(X)
    return s.transform(X)

# #+RESULTS:

#   Note that many of the parameters are supplied with default values.
#   That is no accident.  The values given above, and repeated here, are
#   the ones used throughout this project, and were obtained through
#   experimentation by trial-and-error.

#   |------------------+-----------------+-----------------------------|
#   | Parameter        | Value           | Description                 |
#   |------------------+-----------------+-----------------------------|
#   | =colorspace=     | =COLOR_RGB2HSV= | target color space          |
#   | =channel=        | =2=             | target color channel        |
#   | =orient=         | =9=             | HOG orientation bins        |
#   | =pix_per_cell=   | =8=             | pixels per HOG cell         |
#   | =cell_per_block= | =4=             | cells per HOG block         |
#   | =transform_sqrt= | =False=         | scale values by =math.sqrt= |
#   | =feature_vec=    | =True=          | return feature vector       |
#   |------------------+-----------------+-----------------------------|

#   Next, the ~get_classifier~ function returns a function which is
#   itself a trained classifier.  Parameters control whether or not to
#   train the classifier anew or to load a pre-trained classifier from a
#   file, and what the training/test set split should be when training a
#   new one.

def get_classifier(reload=False,test_size=0.2):
    if reload:
        samples = list(chain(feed("vehicles/**/*.png",1),feed("non-vehicles/**/*.png",0)))  #
        data = cycle(mirror(load(shuffle(samples))))                                        #
        X_train,X_test,y_train,y_test = train_test_split(*zip(*((extract_features(s[0]), s[1]) for s in islice(data, len(samples)))), test_size=test_size, random_state=np.random.randint(0, 100))
        svc = LinearSVC()
        svc.fit(X_train, y_train)
        pickle.dump(svc, open("save.p","wb"))
        print('Test Accuracy of SVC = ', round(classifier.score(X_test, y_test), 4))
    else:
        svc = pickle.load(open("save.p", "rb"))
    return svc

# #+RESULTS:

#   Note the use of our composable utility functions to load the data
#   [[(compose1)][here]] and [[(compose2)][here]].  Note also that there are a variety of classifiers we
#   could use.

#   - [[https://en.wikipedia.org/wiki/Support_vector_machine][Support Vector Machine (SVM)]]
#   - [[https://en.wikipedia.org/wiki/Random_forest][Random Forest]]
#   - [[https://en.wikipedia.org/wiki/Naive_Bayes_classifier][Naive Bayes]]

#   I was prepared to experiment with each of these, and perhaps with
#   their combinations.  I started with an SVG, however, and found that
#   it performed well all on its own.

#   Training a classifier now is as simple as

# classifier = get_classifier(True)

# #+RESULTS:
#   : Test Accuracy of SVC =  0.9938

#   while loading a saved classifier is even simpler

classifier = get_classifier()

# Sliding Window Search I

#   I performed the most experimentation on various sliding window
#   schemes.  Initially, I expended effort on behalf of a single idea:
#   /Can I model vehicle position not in screen coordinates, measured in
#   pixels, but rather in real-world coordinates, measured in meters?/
#   My strategy was to generate sliding windows on a three-dimensional
#   (3D) grid whose origin is where the camera is placed and whose units
#   are meters, and then use geometry to project those windows onto the
#   screen in pixel coordinates.  This model has these assumptions.

#   - The image plane roughly corresponds to the vehicle's windshield.
#   - The windshield is approximately 2 meters wide, 1 meter tall, and 2
#     meters above the road.
#   - The camera is placed approximately 1 meter behind the windshield's
#     center, with a line-of-sight (LOS) perpendicular to it.
#   - The grid coordinates $\left( x, y, z \right)_{grid}$ correspond to
#     the horizontal position across the road, the vertical position
#     above the road, and the longitudinal position down the road.
#   - Positive $x$ values are to the right, negative $x$ values are to
#     the left, and $x_{grid} \in \left[ -15, 15 \right]$.
#   - Negative $y$ values are below the camera, and $y_{grid} \in \left[
#     -2, 0 \right]$.
#   - $z \lt 1$ values are inside the car, and $z_{grid} \in \left[ 10,
#     100 \right]$.

#   These assumptions determine the geometry of the problem and set its
#   physical scale, with a field-of-view (FOV) of 90°, and allow us to
#   create sliding windows as described above.  In principle, vehicle
#   detections on image patches can then be assigned real-world
#   coordinates $(x, y, z)$, or at least road coordinates $(x,
#   z)_{y=0}$, a real-world "heat map" can be built up, and then
#   individual vehicles can be identified either with conventional
#   thresholds + [[http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html][labeling]], with a [[https://en.wikipedia.org/wiki/Blob_detection][Laplace-of-Gaussian]] technique, or
#   with [[https://docs.scipy.org/doc/scipy-0.18.1/reference/cluster.vq.html][/k/-means clustering]].

#   The following coordinate conversion functions support the geometrical model outlined
#   above.

#   - crt2cyl :: Cartesian-to-cylindrical
#   - cyl2crt :: cylindrical-to-Cartesian
#   - cyl2sph :: cylindrical-to-spherical
#   - sph2cyl :: spherical-to-cylindrical
#   - crt2sph :: Cartesian-to-spherical
#   - sph2crt :: spherical-to-Cartesian

#   These are implemented as one-liners

crt2cyl = lambda x,y,z: (math.sqrt(x**2+y**2), math.atan2(y,x), z)
cyl2crt = lambda rho,phi,z: (rho*math.cos(phi), rho*math.sin(phi), z)
cyl2sph = lambda rho,phi,z: (math.sqrt(rho**2+z**2), math.atan2(rho, z), phi)
sph2cyl = lambda r,theta,phi: (r*math.sin(theta), phi, r*math.cos(theta))
crt2sph = lambda x,y,z: (math.sqrt(x**2+y**2+z**2), math.acos(z/math.sqrt(x**2+y**2+z**2)), math.atan2(y,x))
sph2crt = lambda r,theta,phi: (r*math.sin(theta)*math.cos(phi), r*math.sin(theta)*math.sin(phi), r*math.cos(theta))

# #+RESULTS:

#   The ~get_window~ function computes from $(x,y,z)$ a "window", which
#   is a list of tuples wherein the first two provide the corners of its
#   "bounding box" and the last provides the coordinates of its center.
#   Note that it also takes ~height~ and ~width~ parameters for the
#   physical size (in meters) of the window, as well as a ~horizon~
#   parameter, which is the fraction of the image plane (from below) at
#   which the horizon appears.  The default value of ~0.5~ corresponds
#   to the middle.  Finally, like many of my functions it takes a NumPy
#   image array parameter, ~img~, which is mainly for extracting the
#   size and shape of the image.

def get_window(img, x, y, z, horizon=0.5, width=2, height=2):
    d = 1
    r,theta,phi = crt2sph(x,y,z)
    rho2 = d*math.tan(theta)
    x2,y2 = (rho2*math.cos(phi),rho2*math.sin(phi))
    center = (int(img.shape[1]*0.5+x2*img.shape[1]//2),
              int(img.shape[0]*(1-horizon)-y2*img.shape[1]//2))
    scale = img.shape[1]//2
    dx = int(width/2*scale/z)
    dy = int(height/2*scale/z)
    window = [(center[0]-dx,center[1]-dy), (center[0]+dx,center[1]+dy)] + [(x,y,z)]
    return window

# #+RESULTS:

#   Next, the ~draw_window~ function annotates an image ~img~ with the
#   window ~bbox~.  This does not factor into the actual vehicle
#   detection, of course, but the visualization is valuable for
#   understanding how the video processing pipeline ultimately is
#   working.

def draw_window(img, bbox, color=(0,0,255), thick=3):
    cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    return img

# #+RESULTS:

#   For example, first we draw a window box that roughly corresponds to
#   the windshield itself, using a test image from the lecture notes.
#   The windshield's center is at real-world coordinates
#   $(x,y,z)_{windshield} = (0,0,1)$.

image = scale(mpimg.imread("bbox-example-image.jpg"))
draw_window(image, get_window(image, 0, 0.0, 1, horizon=0.5, width=2, height=1))
mpimg.imsave("output_images/windshield.png", image, format="png")

# #+RESULTS:
#   #+begin_example

#   array([[[ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  ..., 
# 	  [ 93, 146, 216],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 98, 153, 220],
# 	  [ 98, 153, 220],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218]],

# 	 ..., 
# 	 [[144, 152, 141],
# 	  [122, 130, 119],
# 	  [109, 117, 106],
# 	  ..., 
# 	  [151, 160, 143],
# 	  [159, 168, 151],
# 	  [159, 168, 151]],

# 	 [[135, 143, 132],
# 	  [136, 144, 133],
# 	  [149, 157, 146],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [162, 171, 154],
# 	  [159, 168, 151]],

# 	 [[130, 138, 127],
# 	  [140, 148, 137],
# 	  [160, 168, 157],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [161, 170, 153],
# 	  [157, 166, 149]]], dtype=uint8)
# #+end_example

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/windshield.png]]

#   Next, we draw window boxes around a few of the cars in the image.
#   Note that here we are eschewing the default value of ~horizon~ in
#   favor of ~0.28~, given the peculiar tilt the camera seems to have in
#   this image.  That value, like the real-world vehicle coordinates
#   $(x,y,z)_i$, were obtained by hand through trial-and-error.

image = scale(mpimg.imread("bbox-example-image.jpg"))
draw_window(image, get_window(image, 4.1, -1.0, 8, horizon=0.28))
draw_window(image, get_window(image, -10.5, -1.0, 22, horizon=0.28))
draw_window(image, get_window(image, -6.1, -1.0, 32, horizon=0.28))
draw_window(image, get_window(image, -0.8, -1.0, 35, horizon=0.28))
draw_window(image, get_window(image, 3, -1.0, 55, horizon=0.28))
draw_window(image, get_window(image, -6.1, -1.0, 55, horizon=0.28))
draw_window(image, get_window(image, -6.1, -1.0, 70, horizon=0.28))
mpimg.imsave("output_images/bbox-example-image-test.png", image, format="png")

# #+RESULTS:
#   #+begin_example

#   array([[[ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  ..., 
# 	  [ 93, 146, 216],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 98, 153, 220],
# 	  [ 98, 153, 220],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218]],

# 	 ..., 
# 	 [[144, 152, 141],
# 	  [122, 130, 119],
# 	  [109, 117, 106],
# 	  ..., 
# 	  [151, 160, 143],
# 	  [159, 168, 151],
# 	  [159, 168, 151]],

# 	 [[135, 143, 132],
# 	  [136, 144, 133],
# 	  [149, 157, 146],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [162, 171, 154],
# 	  [159, 168, 151]],

# 	 [[130, 138, 127],
# 	  [140, 148, 137],
# 	  [160, 168, 157],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [161, 170, 153],
# 	  [157, 166, 149]]], dtype=uint8)
#   array([[[ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  ..., 
# 	  [ 93, 146, 216],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 98, 153, 220],
# 	  [ 98, 153, 220],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218]],

# 	 ..., 
# 	 [[144, 152, 141],
# 	  [122, 130, 119],
# 	  [109, 117, 106],
# 	  ..., 
# 	  [151, 160, 143],
# 	  [159, 168, 151],
# 	  [159, 168, 151]],

# 	 [[135, 143, 132],
# 	  [136, 144, 133],
# 	  [149, 157, 146],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [162, 171, 154],
# 	  [159, 168, 151]],

# 	 [[130, 138, 127],
# 	  [140, 148, 137],
# 	  [160, 168, 157],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [161, 170, 153],
# 	  [157, 166, 149]]], dtype=uint8)
#   array([[[ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  ..., 
# 	  [ 93, 146, 216],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 98, 153, 220],
# 	  [ 98, 153, 220],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218]],

# 	 ..., 
# 	 [[144, 152, 141],
# 	  [122, 130, 119],
# 	  [109, 117, 106],
# 	  ..., 
# 	  [151, 160, 143],
# 	  [159, 168, 151],
# 	  [159, 168, 151]],

# 	 [[135, 143, 132],
# 	  [136, 144, 133],
# 	  [149, 157, 146],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [162, 171, 154],
# 	  [159, 168, 151]],

# 	 [[130, 138, 127],
# 	  [140, 148, 137],
# 	  [160, 168, 157],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [161, 170, 153],
# 	  [157, 166, 149]]], dtype=uint8)
#   array([[[ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  ..., 
# 	  [ 93, 146, 216],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 98, 153, 220],
# 	  [ 98, 153, 220],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218]],

# 	 ..., 
# 	 [[144, 152, 141],
# 	  [122, 130, 119],
# 	  [109, 117, 106],
# 	  ..., 
# 	  [151, 160, 143],
# 	  [159, 168, 151],
# 	  [159, 168, 151]],

# 	 [[135, 143, 132],
# 	  [136, 144, 133],
# 	  [149, 157, 146],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [162, 171, 154],
# 	  [159, 168, 151]],

# 	 [[130, 138, 127],
# 	  [140, 148, 137],
# 	  [160, 168, 157],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [161, 170, 153],
# 	  [157, 166, 149]]], dtype=uint8)
#   array([[[ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  ..., 
# 	  [ 93, 146, 216],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 98, 153, 220],
# 	  [ 98, 153, 220],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218]],

# 	 ..., 
# 	 [[144, 152, 141],
# 	  [122, 130, 119],
# 	  [109, 117, 106],
# 	  ..., 
# 	  [151, 160, 143],
# 	  [159, 168, 151],
# 	  [159, 168, 151]],

# 	 [[135, 143, 132],
# 	  [136, 144, 133],
# 	  [149, 157, 146],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [162, 171, 154],
# 	  [159, 168, 151]],

# 	 [[130, 138, 127],
# 	  [140, 148, 137],
# 	  [160, 168, 157],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [161, 170, 153],
# 	  [157, 166, 149]]], dtype=uint8)
#   array([[[ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  ..., 
# 	  [ 93, 146, 216],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 98, 153, 220],
# 	  [ 98, 153, 220],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218]],

# 	 ..., 
# 	 [[144, 152, 141],
# 	  [122, 130, 119],
# 	  [109, 117, 106],
# 	  ..., 
# 	  [151, 160, 143],
# 	  [159, 168, 151],
# 	  [159, 168, 151]],

# 	 [[135, 143, 132],
# 	  [136, 144, 133],
# 	  [149, 157, 146],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [162, 171, 154],
# 	  [159, 168, 151]],

# 	 [[130, 138, 127],
# 	  [140, 148, 137],
# 	  [160, 168, 157],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [161, 170, 153],
# 	  [157, 166, 149]]], dtype=uint8)
#   array([[[ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  [ 96, 151, 218],
# 	  ..., 
# 	  [ 93, 146, 216],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  [ 97, 152, 219],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218],
# 	  [ 96, 149, 219]],

# 	 [[ 97, 152, 219],
# 	  [ 98, 153, 220],
# 	  [ 98, 153, 220],
# 	  ..., 
# 	  [ 94, 147, 217],
# 	  [ 94, 147, 217],
# 	  [ 95, 148, 218]],

# 	 ..., 
# 	 [[144, 152, 141],
# 	  [122, 130, 119],
# 	  [109, 117, 106],
# 	  ..., 
# 	  [151, 160, 143],
# 	  [159, 168, 151],
# 	  [159, 168, 151]],

# 	 [[135, 143, 132],
# 	  [136, 144, 133],
# 	  [149, 157, 146],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [162, 171, 154],
# 	  [159, 168, 151]],

# 	 [[130, 138, 127],
# 	  [140, 148, 137],
# 	  [160, 168, 157],
# 	  ..., 
# 	  [157, 166, 149],
# 	  [161, 170, 153],
# 	  [157, 166, 149]]], dtype=uint8)
# #+end_example

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/bbox-example-image-test.png]]

#   In order to help visualize the geometry further, we animate a
#   handful of windows receding into the distance.

def zooming_windows(img):
    def make_frame(t):
        frame = np.copy(img)
        z = 2**(t % 5)*5
        draw_window(frame, get_window(frame,-10.5,-1.0,z,horizon=0.28))
        draw_window(frame, get_window(frame,-6.1,-1.0,z,horizon=0.28))
        draw_window(frame, get_window(frame,-0.8,-1.0,z,horizon=0.28))
        draw_window(frame, get_window(frame,4.1,-1.0,z,horizon=0.28))
        cv2.putText(frame, "z: %.2f m" % z, (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
        return frame
    return make_frame

# #+RESULTS:

clip = VideoClip(zooming_windows(mpimg.imread('bbox-example-image.jpg')), duration=5)
clip.write_videofile("output_images/zooming-windows.mp4", fps=25)

# #+RESULTS:
#   : 
#   : [MoviePy] >>>> Building video output_images/zooming-windows.mp4
#   : [MoviePy] Writing video output_images/zooming-windows.mp4
#   :   0% 0/126 [00:00<?, ?it/s]  5% 6/126 [00:00<00:02, 56.66it/s] 14% 18/126 [00:00<00:01, 66.33it/s] 23% 29/126 [00:00<00:01, 75.29it/s] 32% 40/126 [00:00<00:01, 82.77it/s] 38% 48/126 [00:00<00:01, 67.63it/s] 44% 55/126 [00:00<00:01, 48.31it/s] 48% 61/126 [00:00<00:01, 45.15it/s] 53% 67/126 [00:01<00:01, 42.19it/s] 57% 72/126 [00:01<00:01, 41.51it/s] 61% 77/126 [00:01<00:01, 40.84it/s] 66% 83/126 [00:01<00:01, 42.81it/s] 70% 88/126 [00:01<00:00, 43.21it/s] 74% 93/126 [00:01<00:00, 42.52it/s] 78% 98/126 [00:01<00:00, 44.47it/s] 82% 103/126 [00:02<00:00, 40.78it/s] 87% 109/126 [00:02<00:00, 44.94it/s] 90% 114/126 [00:02<00:00, 44.86it/s] 94% 119/126 [00:02<00:00, 44.63it/s] 98% 124/126 [00:02<00:00, 44.07it/s] 99% 125/126 [00:02<00:00, 50.83it/s]
#   : [MoviePy] Done.
#   : [MoviePy] >>>> Video ready: output_images/zooming-windows.mp4

#   #+HTML: <iframe width="800" height="450" src="https://www.youtube.com/embed/lqp9rOSPVrc" frameborder="0" allowfullscreen></iframe>

#   This is just for visualization.  For vehicle detection, a denser
#   grid should be used, and we raster the windows horizontally as they
#   ratchet down-range.  We also confine the windows to a horizontal
#   plane, at $z = -1$.  But, because this sliding window and other ones
#   like it actually will be used in the vehicle-detection
#   video-processing pipeline, it is worthwhile to remove windows that
#   exceed the image boundary.  That is the purpose of the ~clip_window~
#   function.

def clip_window(x, box):
    return sum([box[0]<=x[0][0]<=box[1],
                box[0]<=x[1][0]<=box[1],
                box[2]<=x[0][1]<=box[3],
                box[2]<=x[1][1]<=box[3]])==4

# #+RESULTS:

  
#   Since our strategy will be to write functions to produce "grids"
#   that can be used both for visualization and for vehicle-detection,
#   we refactor much of the animated visualization into a new function,
#   ~get_frame_maker~.

def get_frame_maker(img, grid):
    def make_frame(t):
        frame = np.copy(img)
        draw_window(frame, grid.__next__()[:2], color=(0,255,255))
        return frame
    return make_frame

# #+RESULTS:

#   With these tools, first we define a "sparse grid"

def sparse_scan(img):
    grid = np.mgrid[-15:15:2,-1.0:0:2,3:7:1]
    grid[2,]=2**grid[2,]
    grid = grid.T.reshape(-1,3)
    grid = (get_window(img,x[0],x[1],x[2], horizon=0.28)+[x] for x in grid)
    grid = filter(lambda x: clip_window(x, (0, img.shape[1], (img.shape[0]//2), img.shape[0])), grid)
    return grid

# #+RESULTS:

#   visualize its 40 windows

image = scale(mpimg.imread("bbox-example-image.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), sparse_scan(image)))))
mpimg.imsave("output_images/sparse-scan.png", image, format="png")

# #+RESULTS:
#   : 
#   : 52

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/sparse-scan.png]]

#   and then animate them.

clip = VideoClip(get_frame_maker(image, cycle(sparse_scan(image))), duration=10)
clip.write_videofile("output_images/sparse-scan.mp4", fps=25)

# #+RESULTS:
#   : 
#   : [MoviePy] >>>> Building video output_images/sparse-scan.mp4
#   : [MoviePy] Writing video output_images/sparse-scan.mp4
#   :   0% 0/251 [00:00<?, ?it/s]  1% 2/251 [00:00<00:17, 14.44it/s]  5% 13/251 [00:00<00:12, 19.45it/s] 10% 25/251 [00:00<00:08, 25.89it/s] 15% 37/251 [00:00<00:06, 33.67it/s] 19% 47/251 [00:00<00:05, 38.57it/s] 22% 54/251 [00:00<00:05, 37.23it/s] 24% 60/251 [00:00<00:04, 39.07it/s] 26% 66/251 [00:01<00:04, 39.37it/s] 29% 73/251 [00:01<00:04, 42.72it/s] 31% 79/251 [00:01<00:03, 44.27it/s] 34% 85/251 [00:01<00:03, 45.24it/s] 36% 91/251 [00:01<00:03, 47.37it/s] 39% 97/251 [00:01<00:03, 46.74it/s] 41% 102/251 [00:01<00:03, 46.19it/s] 43% 107/251 [00:01<00:03, 46.56it/s] 45% 112/251 [00:02<00:03, 43.55it/s] 47% 118/251 [00:02<00:02, 46.99it/s] 49% 124/251 [00:02<00:02, 47.81it/s] 51% 129/251 [00:02<00:02, 44.38it/s] 53% 134/251 [00:02<00:02, 44.91it/s] 55% 139/251 [00:02<00:02, 44.91it/s] 58% 145/251 [00:02<00:02, 47.61it/s] 60% 151/251 [00:02<00:02, 44.84it/s] 63% 157/251 [00:03<00:01, 48.51it/s] 65% 163/251 [00:03<00:01, 47.18it/s] 67% 168/251 [00:03<00:01, 46.96it/s] 69% 173/251 [00:03<00:01, 47.15it/s] 71% 178/251 [00:03<00:01, 46.23it/s] 73% 183/251 [00:03<00:01, 45.98it/s] 75% 189/251 [00:03<00:01, 47.93it/s] 77% 194/251 [00:03<00:01, 46.38it/s] 79% 199/251 [00:03<00:01, 46.40it/s] 82% 205/251 [00:04<00:00, 48.70it/s] 84% 210/251 [00:04<00:00, 44.65it/s] 86% 216/251 [00:04<00:00, 45.64it/s] 88% 221/251 [00:04<00:00, 46.14it/s] 90% 226/251 [00:04<00:00, 47.07it/s] 92% 231/251 [00:04<00:00, 46.55it/s] 94% 237/251 [00:04<00:00, 48.92it/s] 97% 243/251 [00:04<00:00, 50.18it/s] 99% 249/251 [00:05<00:00, 45.05it/s]100% 250/251 [00:05<00:00, 49.80it/s]
#   : [MoviePy] Done.
#   : [MoviePy] >>>> Video ready: output_images/sparse-scan.mp4

#   #+HTML: <iframe width="800" height="450" src="https://www.youtube.com/embed/Vn1HxPRd2W0" frameborder="0" allowfullscreen></iframe>

#   We can also define a "dense grid" with more windows, scanning the
#   roadway with finer resolution in the $x$ and $z$ directions.  We
#   skip the animation this time, as it is rather boring.

def dense_scan(img, h=2,w=2):
    grid = np.mgrid[-15:15:0.5,-1.0:0:2,10:100:2]
    grid = grid.T.reshape(-1,3)
    grid = (get_window(img,x[0],x[1],x[2], horizon=0.28, height=h, width=w)+[x] for x in grid)
    grid = filter(lambda x: clip_window(x, (0, img.shape[1], (img.shape[0]//2), img.shape[0])), grid)
    return grid

# #+RESULTS:

#   When produce the grid image, note that it has 2600+ windows!  That
#   probably is excessive and would slow down video processing.

image = scale(mpimg.imread("bbox-example-image.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), dense_scan(image)))))
mpimg.imsave("output_images/dense-scan.png", image, format="png")

# #+RESULTS:
#   : 
#   : 2653

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/dense-scan.png]]

#   The sparse scan above probably is too sparse, but one way we can
#   reduce the number of windows would be to search the perimeter of the
#   road, where new cars are likely to come on-stage.

def perimeter_scan(img):
    grid = np.mgrid[-15:15:0.5,-1.0:0:2,10:100:2]
    grid = grid.T.reshape(-1,3)
    grid = list(filter(lambda x: not (-4<=x[0]<=4 and 5<=x[2]<=40), grid))
    grid = (get_window(img,x[0],x[1],x[2], horizon=0.28)+[x] for x in grid)
    grid = filter(lambda x: clip_window(x, (0, img.shape[1], (img.shape[0]//2), img.shape[0])), grid)
    return grid

# #+RESULTS:

image = scale(mpimg.imread("bbox-example-image.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), perimeter_scan(image)))))
mpimg.imsave("output_images/perimeter-scan.png", image, format="png")

# #+RESULTS:
#   : 
#   : 2381

#   Sadly, this barely makes a dent in reducing the number of windows.  

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/perimeter-scan.png]]

#   In order to make headway, a simple choice is just to stick with the
#   dense grid, perform vehicle detections with it against a test image,
#   and gauge its performance.

#   To do that, we need a function ~get_patches~, that takes a /window/,
#   which again is mainly a bounding-box (with pixel dimensions) into a
#   /patch/, which is a NumPy image sub-array taken from a larger image.

def get_patches(img, grid, size=(64,64)):
    return ((cv2.resize(img[window[0][1]:window[1][1],
                            window[0][0]:window[1][0]],size),window) for window in grid)

# #+RESULTS:

#   Armed with that function, next we just map our classifier over all
#   of the window patches on an image.

def process(x):
    return (classifier.predict(extract_features(x[0]))[0],x[1])

# #+RESULTS:

results = list(map(process, get_patches(image, dense_scan(image))))
print(len(results))

# #+RESULTS:
#   : 
#   : 2653

#   To visualize where vehicle detections have occurred on our dense grid
#   over the road, we filter the processed results that have a value
#   greater than 1 (i.e., a detection has occurred for that window patch)

_,r = zip(*filter(lambda x: x[0]>0, results))
_,_,cen,_ = zip(*r)
x,y,z = zip(*cen)
plt.scatter(x,z,s=50,c=y)

# Sliding Window Search II

#   To refresh the reader, a more traditional sliding windows approach
#   models a grid of windows and their image patches not in real-world
#   3D physical space, but in 2D image space.  This involves trade-offs.
#   On the one hand, we give up a straightforward 3D interpretation of a
#   vehicle detection event.  In principle, we could still recover
#   distance information by deprojecting the window (the reverse of our
#   operation above), but at the expense of greater complication.  On
#   the other hand, we gain with this trade-off a simpler implementation
#   that already has a proven track-record.

#   We can reuse much of our other code, though, since we just need to
#   define functions to produce grids that obey whatever selection
#   functions we desire.

#   First up is a simple "image-plane scan", which carpets the image
#   plane in a uniform grid of windows at various fixed scales.

def image_plane_scan(img,ny,overlap,scale):
    size = int(img.shape[0]//ny)//scale
    delta = int(size*(1-overlap))
    box1 = (0,
            img.shape[1],
            (img.shape[0]-img.shape[0]//scale)//2,
            img.shape[0] - (img.shape[0]-img.shape[0]//scale)//2)
    box2 = (0,
            img.shape[1],
            (img.shape[0]//2),
            img.shape[0])
    grid = np.mgrid[0:img.shape[1]:delta,
                    img.shape[0]:-delta:-delta].T.reshape(-1,2)
    grid = ([(c[0],c[1]), (c[0]+size,c[1]+size)] for c in grid)
    grid = filter(lambda x: clip_window(x, box1), grid)
    grid = filter(lambda x: clip_window(x, box2), grid)
    return grid

# #+RESULTS:

image = scale(mpimg.imread("test_images/test1.jpg"))
print()
print("Number of windows: %s" %
      len(list(map(lambda w: draw_window(image, w[:2]),
                   chain(
                       image_plane_scan(image,4,0.50,1),
                       image_plane_scan(image,4,0.50,2),
                       image_plane_scan(image,4,0.50,3)
                   )))))
mpimg.imsave("output_images/imageplane-scan.png", image, format="png")

# #+RESULTS:
#   : 
#   : 
#   : ... ... ... ... ... ... Number of windows: 243

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/imageplane-scan.png]]

#   This produces 1400+ images, which highlights a persistent problem I
#   grappled with.  There is an inherent trade-off between the accuracy
#   of a dense window sample, and the performance of a sparse window sample.

#   A conjecture I had to help ease the tension between these two poles
#   was to relax the constraint of a regular grid of windows in favor of
#   a random scattering of windows.  One of the reasons the window count
#   soared with a regular grid was the overlap; a high degree of overlap
#   (>50%) was needed for higher spatial resolution of detected vehicle
#   locations, but the number of windows is essentially quadratic in the
#   degree of overlap.  However, the stochastic behavior of an irregular
#   random sampling of windows means that a higher spatial resolution
#   can be achieved in an economy of window patches.  

#   The trade-offs here, however, are two-fold.  First, we no longer can
#   pre-compute the grid, but instead must compute a new random ensemble
#   of windows for each video frame.  In the testing that I did, this
#   proved to be of little concern; the Python profiler, and experience
#   as well, showed that the grid computation time was relatively
#   trivial.  The bulk of the time was spent on feature extraction and
#   classification for each window patch, a task that obviously cannot
#   be precomputed irrespective of the grid strategy.

#   Second, since any one /particular/ frame is treated with a
#   relatively sparse (but now random) irregular grid of windows, this
#   intensifies the need for integrating the signal over multiple frames
#   (a task we anticipated in any case).  Consequently, we lose
#   resolution in the time domain.  While that could be a problem for
#   fast-moving vehicles, it was not for the relatively slow relative
#   velocity of the vehicles in our project video.

#   My first version of a random scan uses a region-of-interest mask
#   that selects out a trapezoidal region covering just the border of
#   the road.

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# #+RESULTS:

#   The actual function ~random_scan~ takes an image ~img~ (again, just
#   for the size information), and a window size.  Since these we are
#   now operating in the pixel coordinates of the image plane rather
#   than in the physical coordinates of the real world, the window size
#   is taken just in pixels.  This function works by thresholding a
#   random array.  It is a somewhat elegant technique, but is
#   inefficient and /slowww/.

def random_scan(img,size):
    x = np.random.rand(*img.shape[:2])
    x[x<0.999] = 0
    x = scale(np.ceil(x))
    x = region_of_interest(x, np.array([[[0, 0.5*image.shape[0]],
                           [image.shape[1], 0.5*image.shape[0]],
                           [image.shape[1], image.shape[0]],
                           [(1-2/6)*image.shape[1], 0.5*image.shape[0]],
                           [(2/6)*image.shape[1], 0.5*image.shape[0]],
                           [0, image.shape[0]]]]).astype('int'))
    x = np.dstack(np.nonzero(x))
    s = np.random.choice(2**np.arange(4), len(x[0]))
    grid = ([(c[1],c[0]),
             (c[1]+size,c[0]+size)] for c in x[0])
    return grid

# #+RESULTS:

image = scale(mpimg.imread("test_images/test1.jpg"))
print()
print("Number of windows: %s" %
      len(list(map(lambda w: draw_window(image, w[:2]),
                   chain(
                       random_scan(image,180),
                       random_scan(image,90),
                       random_scan(image,60)
                   )))))
mpimg.imsave("output_images/random-scan1.png", image, format="png")

# #+RESULTS:
#   : 
#   : 
#   : ... ... ... ... ... ... Number of windows: 465

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/random-scan1.png]]

#   The next random grid function ~random_scan2~, uses a slightly
#   less-elegant approach, but is noticeably faster.  Aside from
#   confining the window to the bottom half of the image, however, it
#   does not use a region-of-interest mask.

def random_scan2(img,size,num=100):
    x = np.random.rand(num,2)
    x[:,0]*=image.shape[1]
    x[:,1]*=image.shape[1]
    x = x.astype('int')
    x = x[x[:,1]<image.shape[0]]
    x = x[x[:,1]>=image.shape[0]//2]
    box = (0,img.shape[1],(img.shape[0]//2),650)
    grid = ([(c[0],c[1]),
             (c[0]+size,c[1]+size)] for c in x)
    grid = filter(lambda x: clip_window(x, box), grid)
    return grid

# #+RESULTS:

image = scale(mpimg.imread("test_images/test1.jpg"))
print("Number of windows: %s" %
      len(list(map(lambda w: draw_window(image, w[:2]),
                   chain(
                       random_scan2(image,256,1000),
                       random_scan2(image,128,1000),
                       random_scan2(image,64,1000)
                   )))))
mpimg.imsave("output_images/random-scan2.png", image, format="png")

# #+RESULTS:
#   : 
#   : ... ... ... ... ... ... Number of windows: 311

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/random-scan2.png]]

#   The next random scanner I tried worked in polar (pixel) coordinates
#   so as to achieve a masking affect, that concentrates windows on the
#   road borders where vehicles are most likely to appear.

def random_scan3(img,size,num=100,minr=None,maxr=None,mintheta=None,maxtheta=None,center=None,scale=True):
    if center==None:
        center = tuple(np.array(image.shape[:2][::-1])//2)
    polar = np.random.rand(num,2)
    polar[:,0]*=image.shape[1]
    polar[:,1]*=math.pi*2
    if not minr==None:
        polar = polar[polar[:,0]>=minr]
    if not maxr==None:
        polar = polar[polar[:,0]<maxr]
    if not mintheta==None:
        polar = polar[polar[:,1]>=0]
    if not maxtheta==None:
        polar = polar[polar[:,1]<maxtheta]
    if scale:
        s = (size//2*polar[:,0]/image.shape[1]).astype('int')
    else:
        try:
            dist = int(math.sqrt(sum([(center[0]-image.shape[1]//2)**2,
                                      (center[1]-image.shape[0]//2)**2])))
            s = [int(size*(dist/(image.shape[1]//2)))]*len(polar)
        except:
            pdb.set_trace()
    x,y=zip(*np.dstack((center[0]+polar[:,0]*np.cos(polar[:,1]),
                        center[1]+polar[:,0]*np.sin(polar[:,1]))).astype('int')[0])
    grid = ([(c[0]-c[2],c[1]-c[2]), (c[0]+c[2],c[1]+c[2])] for c in zip(x,y,s))
    box = (0,img.shape[1],(0),670)
    grid = filter(lambda x: clip_window(x, box), grid)
    return grid

# #+RESULTS:

image = scale(mpimg.imread("test_images/test1.jpg"))
print("Number of windows: %s" %
      len(list(map(lambda w: draw_window(image, w[:2]),
                   chain(
                       random_scan3(image,image.shape[1]//4,
                                    3000,
                                    minr=image.shape[0]//3,
                                    mintheta=0,
                                    maxtheta=math.pi)
                   )))))
mpimg.imsave("output_images/random-scan3.png", image, format="png")

# #+RESULTS:
#   : 
#   : ... ... ... ... ... ... ... ... Number of windows: 199

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/random-scan3.png]]

#   This produces an interesting pattern, but I was not comfortable
#   peculiar way the windows are scaled to different sizes, so I wrote
#   yet another grid window function ~random_scan4~, which is a bit of a
#   hybrid.  It actually re-uses the 3D model described above in
#   *Sliding Window Search I*.  Windows are defined in a 3D volume which
#   covers the road from left to right, from the car to the horizon, and
#   from the camera level down to the road.  I.e., it is like a long,
#   thick "ribbon", within which windows are randomly sampled.  As
#   above, we are back in physical space for window sizes, rather than
#   in pixel space.  Finally, the physical space windows are projected
#   back onto the image plane to give us a grid window in pixel-space.
#   In fact, this is almost exactly as we did in the earlier section.
#   The main differences are:

#   1. We discard the 3D window location information after projecting it
#      to a grid window on the image plane.
#   2. Window locations are randomly drawn from the 3D volume described
#      above, rather than laid out in a regular array.

#   This is implemented in the ~random_scan4~ function.

def random_scan4(img,size,num=100,width=25,left=-12.5):
    grid = np.random.rand(num,3)
    grid[:,0]*=width
    grid[:,1]*=2
    grid[:,1]-=2
    grid[:,2]*=40
    grid[:,0]+=left
    grid[:,1]-=4
    grid[:,2]+=5
    grid = grid.astype('int')
    grid = (get_window(img,x[0],x[1],x[2], height=4, width=4)+[x] for x in grid)
    grid = filter(lambda x: clip_window(x, (0, img.shape[1], (img.shape[0]//2), img.shape[0])), grid)
    return grid

# #+RESULTS:

image = scale(mpimg.imread("test_images/test1.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), random_scan4(image,2,1000)))))
mpimg.imsave("output_images/random-scan4.png", image, format="png")

# #+RESULTS:
#   : 
#   : 813

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/random-scan4.png]]

#   Note that the above function takes parameters =width= and =left=
#   which set the width of the "ribbon" volume, and its left edge (in
#   meters).  We can easily combine a couple calls to this grid
#   generating function with judicious parameter choices in order to
#   archive interesting search patterns.  For instance, in
#   ~random_scan5~, we superimpose two ribbons, one on the left, and one
#   on the right, in order just to search the road borders.

def random_scan5(img,size,num=100):
    grid = chain(random_scan4(img,size,num//2,width=20,left=-30),
                 random_scan4(img,size,num//2,width=20,left=+10))
    return grid

# #+RESULTS:

image = scale(mpimg.imread("test_images/test1.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), random_scan5(image,2,1000)))))
mpimg.imsave("output_images/random-scan5.png", image, format="png")

# Video Implementation

#   With a variety of strategies for searching a video frame for vehicle
#   detection events, the next major step is to adapt those strategies
#   into an implementation of a video-processing pipeline.  The
#   processing pipeline has these major steps.

#   1. Get a frame of video.
#   2. Generate a grid of windows using one of the schemes developed
#      above.
#   3. Using the frame image and the grid of windows, generate a
#      sequence of patches, which are small (64 x 64 pixel) sub-arrays,
#      compatible with our classifier.
#   4. Apply the classifier as a stencil over the sequence of patches to
#      generate a sequence of detection/non-detection events.
#   5. Assign a positive value (1) to the bounding box associated with
#      each window/patch, and superimpose these detection patches to
#      create a 2D histogram, which we'll call a "heat map."
#   6. Apply a threshold to the heat map for each frame, by selecting
#      only those array values that exceed the threshold.
#   7. Use the [[http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label][=label=]] function to identify connected regions in the
#      thresholded heat map, and associate these with "vehicles."
#   8. Annotate the original frame image with a bounding box for each
#      vehicle.

#   The previous sections were largely the province of Steps 2, 3, and 4
#   above.  Picking up from there with step 5, we need to build up a
#   heat map.  The ~add_heat~ function does just that.  It takes
#   single-channel image array ~heatmap~ (typically, all zeros) and a
#   list of window/patches in ~bbox_list~, and builds up the
#   histogram/heat map.

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1],
                box[0][0]:box[1][0]] += 1
    return heatmap

# #+RESULTS:

#   For step 6, we add the function ~apply_threshold~, which selects out
#   of a heatmap image array ~heat~ only those array elements that
#   exceed ~threshold~.

def apply_threshold(heat, threshold):
    heatmap = np.copy(heat)
    heatmap[heatmap <= threshold] = 0
    return heatmap

# #+RESULTS:

#   Steps 7 and 8 are combined in the next function,
#   ~draw_labeled_boxes~, which takes a multi-channel image array
#   (typically, the original frame of video) in ~img~, along with
#   labeled regions in ~labels~, computes the locations and bounding
#   boxes for detected vehicles, and annotates the frame image with a
#   box.

def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        center = (int(np.mean((np.min(nonzerox), np.max(nonzerox)))),
                  int(np.mean((np.min(nonzeroy), np.max(nonzeroy)))))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        cv2.putText(img, "Car: %s" % car_number,
                    (bbox[0][0],bbox[0][1]-20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img, "Center: %s" % (center,),
                    (bbox[0][0],bbox[0][1]-10),
                    cv2.FONT_HERSHEY_DUPLEX, .5, (255,255,255), 1)
    return img

# #+RESULTS:

#   In order to test these functions on a single image, we define a
#   simple ~process~ function, which does feature-extraction and
#   classification using the ~extract_features~ function and the
#   ~classifier~ we trained earlier.

def process(x):
    return (classifier.predict(extract_features(x[0]))[0],x[1])

# #+RESULTS:

#   For this test, we choose as our grid strategy just the simple
#   ~image_plane_scan~ defined above.  Recall that this function covers
#   an image with regular arrays of overlapping windows at fixed scales,
#   and that it trades performance for simplicity.  The fact that it
#   generates many windows and therefore operates slowly is of no
#   concern for just one image.

image = scale(mpimg.imread("test_images/test1.jpg"))
grid = list(chain(
    image_plane_scan(image,4,0.750,1),
    image_plane_scan(image,4,0.750,2),
    image_plane_scan(image,4,0.750,3),
))
results = map(process, get_patches(image, grid))
image = scale(mpimg.imread("test_images/test1.jpg"))
box_list = list(map(lambda x: x[1][:2], filter(lambda x: x[0]>0, results)))
heat = np.zeros_like(image[:,:,0]).astype(np.float)
heat = add_heat(heat,box_list)
heat = apply_threshold(heat,5)
labels = label(heat)
print(labels[1], 'cars found')
draw_img = draw_labeled_bboxes(np.copy(image), labels)
fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heat, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.savefig("output_images/heatmaptest.png")

# #+RESULTS:
#   : 
#   : ... ... ... ... >>> >>> >>> >>> >>> >>> >>> >>> 2 cars found
#   : >>> >>> <matplotlib.axes._subplots.AxesSubplot object at 0x7f8fb8519d30>
#   : <matplotlib.image.AxesImage object at 0x7f8fbbde4c50>
#   : <matplotlib.text.Text object at 0x7f8fb808f320>
#   : <matplotlib.axes._subplots.AxesSubplot object at 0x7f8fbbdea2b0>
#   : <matplotlib.image.AxesImage object at 0x7f8fbbdea278>
#   : <matplotlib.text.Text object at 0x7f8fbbe0af98>

#   #+ATTR_HTML: :width 800px
#   [[file:output_images/heatmaptest.png]]

#   Satisfied that we are acquiring the pieces we need, we are almost
#   ready to create our actual video-processing pipeline.  Before doing
#   that, however, we must address the related topics of
#   false-positives, buffering, smoothing, moving averages, and
#   time-space resolution.

#   In the video-processing pipeline, the raw signal will be coming at
#   us at a full 25 frames per second, with a lot of noise associated
#   with highly-transient "false-positive" detection events at locations
#   where there is no vehicle.  A tried-and-true way of coping with this
#   is to buffer the signal over some time interval (which corresponds
#   to some number of frames), integrate the signal over that buffer,
#   and substitute a (possibly weighted) moving average over the buffer
#   for the raw signal.  This smooths and conditions the signal,
#   largely eliminating false-positive detections and "jitter", though
#   it comes at a small price.  Our time-domain resolution dilates from
#   the raw frame-rate to the time interval associated with our
#   smoothing buffer.  Consequently, we should take care to make our
#   buffer "long enough, and no longer"; it should remove most false
#   positives, but still distinguish different cars, for instance.  

#   One way to implement smoothing is to introduce a bona-fide
#   frame-buffer, such as with a [[https://en.wikipedia.org/wiki/Circular_buffer][ring buffer]] like Python's [[https://docs.python.org/2/library/collections.html#collections.deque][~deque~]]
#   data-structure.  

#   However, it turns out that is not at all necessary.

#   Instead, we can embellish the analogy we have adopted of a "heat
#   map" with the idea of "cooling."  First, we move the heat map data
#   structure (typically, a 1-channel 2-D image array of the same size
#   as a video frame) /outside/ of the main processing loop.  A global
#   variable, a local variable in a [[https://www.programiz.com/python-programming/closure][closure]], or a class object's member
#   variable all are good candidates here.  Then, in our main processing
#   loop, for each video frame we deliver a "heat pulse" into the heat
#   map, through the ~add_heat~ map function above, for instance.  By
#   itself, that would integrate the signal over the whole video.  To
#   recover the notions of a short-interval buffer, a moving average,
#   and a smoothing kernel, we just insert a "cooling" step before the
#   heat pulse.  I.e., we "cool" the heat map by dis-integrating some of
#   its accumulated signal.  A simple way to do that is just to
#   down-scale it by a small /cooling factor/---say 1%---by multiplying
#   it by a fraction.

#   In fact, doing this imitates none other than one of the simplest and
#   oldest-known cooling laws in Nature, [[https://en.wikipedia.org/wiki/Newton's_law_of_cooling][Newton's Law of Cooling]].  This
#   is the cooling law for conductive media, and corresponds precisely
#   to [[https://en.wikipedia.org/wiki/Exponential_smoothing][exponential smoothing]]

#   \[ \Delta T (t) = \Delta T (0) e^{-t/t_0} \]

#   where $t_0$ is the cooling time-scale and relates to the cooling
#   factor.  Just as a guide, the following table lists the approximate
#   time for the heat map to cool by one-half, at the video's 25 frames
#   per second, for a handful of cooling factors.

#   |----------------+--------------------|
#   | Cooling Factor | Cooling Time-Scale |
#   |----------------+--------------------|
#   |           0.99 | 3 seconds          |
#   |           0.98 | 1.5 seconds        |
#   |           0.97 | 1 second           |
#   |           0.95 | 0.5 second         |
#   |           0.90 | 10 milliseconds    | 

#   This is the buffering strategy we use here, and it works quite
#   well.  I decide to go for an object-oriented approach, so I made the
#   heat map into an object member variable.

#   The ~Component~ class defines that object, along with a host of
#   other attributes and methods for performing the video processing.
#   Hyper-parameters that govern the operation of the pipeline are
#   injected via the object constructor, and that includes the
#   ~cooling_factor~ described above.  Its default value is 0.98, for a
#   cooling time of between 1 and 2 seconds.  Note that most of the
#   class functions really are just thin wrappers over the functions we
#   developed above.  The main exception to that is the ~get_out_img~
#   function, which composes the "scene" for the video frame.  It
#   supports an "extended" branch of operation wherein the main video
#   window is supplemented with smaller windows animating the grid
#   windows, and the evolving heat map.  In its layout, I made room for
#   5 such smaller windows, though in the end I only needed 2 of them.

class Component:
    def __init__(self, img,
                 cell_per_block = 4,
                 channel = 2,
                 colorspace = cv2.COLOR_RGB2HSV,
                 feature_vec = True,
                 orient = 9,
                 pix_per_cell = 8,
                 transform_sqrt = False,
                 test_size = 0.2,
                 threshold = 25,
                 numwindows = 100,
                 cooling_factor = 0.98,
                 center=None,
                 extended=True,
                 size=None):
        self.bboxwindow = np.copy(image)
        self.cell_per_block = cell_per_block
        self.center = center if center else tuple(np.array(img.shape[:2][::-1])//2)
        self.channel = channel
        self.children = []
        self.colorspace = colorspace
        self.cooling_factor = cooling_factor
        self.feature_vec = feature_vec
        self.flat = np.zeros_like(image[:,:,0]).astype(np.float)
        self.heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        self.image = img
        self.labels = None
        self.mainwindow = np.copy(image)
        self.numwindows = numwindows
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.size = size if size else min(img.shape[:2])//2
        self.test_size = test_size
        self.threshold = threshold
        self.transform_sqrt = transform_sqrt
        self.extended = extended
 
 
    def get_center(self):
        return self.center
 
 
    def get_size(self):
        return self.size
 
 
    def cool(self):
        self.heatmap*=self.cooling_factor
 
 
    def get_heatmap(self):
        return self.heatmap
 
 
    def sample(self, mainwindow, grid):
        results = map(process, get_patches(mainwindow, grid))
        return results
 
 
    def heat(self, results):
        samples = list(map(lambda x: x[1][:2], filter(lambda x: x[0]>0, results)))
        for s in samples:
            self.heatmap[s[0][1]:s[1][1],
                         s[0][0]:s[1][0]] += 1
 
 
    def evolve(self, image):
        self.cool()
        self.mainwindow = np.copy(image)
        self.bboxwindow = np.copy(image)
        self.chld_img = np.dstack([self.flat, self.flat, self.flat])
        grid = self.grid(self.numwindows)
        self.addboxes(self.bboxwindow, grid)
        results = self.sample(self.mainwindow, grid)
        self.heat(results)
        self.heatmap = cv2.GaussianBlur(self.heatmap, (31, 31), 0)
        thresholded = apply_threshold(self.get_heatmap(),self.threshold)
        self.labels = label(thresholded)
        draw_labeled_bboxes(self.mainwindow, self.labels)
 
 
    def get_out_img(self):
        if self.extended:
            bbox_img = cv2.resize(self.bboxwindow, tuple(np.array(self.image.shape[:2][::-1])//2))
            hot2_img = cv2.resize(scale(np.dstack([self.get_heatmap(), self.get_heatmap(), self.flat]), 2*self.threshold), tuple(np.array(image.shape[:2][::-1])//2))
            cv2.putText(hot2_img, "Max: %.2f" % np.max(self.get_heatmap()), (25,25), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
            cv2.putText(hot2_img, "Threshold: %.2f" % np.max(self.threshold), (25,55), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
            cv2.putText(hot2_img, "Cooling Fac.: %.2f" % np.max(self.cooling_factor), (25,85), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
            flat_img = cv2.resize(np.dstack([self.flat, self.flat, self.flat]), tuple(np.array(image.shape[:2][::-1])//2))
            outp_img = cv2.resize(np.hstack((np.vstack((self.mainwindow,
                                                        np.hstack((flat_img,
                                                                   flat_img)))),
                                             np.vstack((bbox_img,
                                                        hot2_img,
                                                        flat_img)))),
                                  tuple(np.array(self.image.shape[:2][::-1])))
        else:
            outp_img = self.mainwindow
        return outp_img
 
 
    def grid(self, num):
        return list(random_scan4(self.image, 2, num,width=60,left=-30))
        return list(random_scan5(self.image, 2, num))
 
 
    def addboxes(self, bboxwindow, grid):
        list(map(lambda w: draw_window(bboxwindow, w[:2]), grid))
 
 
    def process_image(self, image):
        self.evolve(image)
        return self.get_out_img()

# #+RESULTS:

#   First, we use the ~Component~ class to process the test video.

in_clip = VideoFileClip("test_video.mp4")
scene = Component(scale(mpimg.imread("test_images/test1.jpg")),threshold=20,cooling_factor=0.99)
out_clip = in_clip.fl_image(scene.process_image)
out_clip.write_videofile("output_images/test_output.mp4", audio=False)

# #+RESULTS:
#   | array | ((-8.5 -1 10)) | array | ((-7 -1 10)) | array | ((-6.5 -1 10)) | array | ((-5.5 -1 10)) | array | ((-5 -1 10)) | array | ((-4.5 -1 10)) | array | ((-3 -1 10)) | array | ((-0.5 -1 10)) | array | ((2.5 -1 10)) | array | ((4.5 -1 10)) | array | ((6.5 -1 10)) | array | ((8 -1 10)) | array | ((-10 -1 12)) | array | ((-9.5 -1 12)) | array | ((-8.5 -1 12)) | array | ((-0.5 -1 12)) | array | ((2.5 -1 12)) | array | ((3 -1 12)) | array | ((6.5 -1 12)) | array | ((9.5 -1 12)) | array | ((-12 -1 14)) | array | ((-11 -1 14)) | array | ((-10 -1 14)) | array | ((-9.5 -1 14)) | array | ((-8.5 -1 14)) | array | ((-0.5 -1 14)) | array | ((9 -1 14)) | array | ((11 -1 14)) | array | ((-13 -1 16)) | array | ((-12.5 -1 16)) | array | ((-7.5 -1 16)) | array | ((2.5 -1 16)) | array | ((-13 -1 18)) | array | ((-8 -1 18)) | array | ((-0.5 -1 18)) | array | ((5.5 -1 18)) | array | ((10 -1 18)) | array | ((10.5 -1 18)) | array | ((14 -1 18)) | array | ((-14.5 -1 20)) | array | ((-14 -1 20)) | array | ((3 -1 20)) | array | ((9 -1 20)) | array | ((9.5 -1 20)) | array | ((10.5 -1 20)) | array | ((11 -1 20)) | array | ((13 -1 20)) | array | ((13.5 -1 20)) | array | ((-13 -1 22)) | array | ((-11 -1 22)) | array | ((-10.5 -1 22)) | array | ((11.5 -1 22)) | array | ((14 -1 22)) | array | ((14.5 -1 22)) | array | ((-14.5 -1 24)) | array | ((-11.5 -1 24)) | array | ((3 -1 24)) | array | ((12.5 -1 24)) | array | ((14 -1 24)) | array | ((-12.5 -1 26)) | array | ((-2.5 -1 26)) | array | ((3 -1 26)) | array | ((4 -1 26)) | array | ((13.5 -1 26)) | array | ((-15 -1 28)) | array | ((-13.5 -1 28)) | array | ((-2.5 -1 28)) | array | ((13 -1 28)) | array | ((14.5 -1 28)) | array | ((-14.5 -1 30)) | array | ((-15 -1 32)) | array | ((3.5 -1 38)) |

#   The test video is very short, so there barely is enough time for
#   "heat to build up" to the point that one car is detected, let alone
#   two.  

#   #+HTML: <iframe width="800" height="450" src="https://www.youtube.com/embed/McviDE-LWLA" frameborder="0" allowfullscreen></iframe>

#   Next, we process the main project video in "extended mode", to
#   include the smaller sub-windows the the animated grid windows and
#   evolving heat map.

in_clip = VideoFileClip("project_video.mp4")
scene = Component(scale(mpimg.imread("test_images/test1.jpg")),threshold=40,cooling_factor=0.99)
out_clip = in_clip.fl_image(scene.process_image)
out_clip.write_videofile("output_images/project_output_extended.mp4", audio=False)

# #+RESULTS:
#   | array | ((-8.5 -1 10)) | array | ((-7 -1 10)) | array | ((-6.5 -1 10)) | array | ((-5.5 -1 10)) | array | ((-5 -1 10)) | array | ((-4.5 -1 10)) | array | ((-3 -1 10)) | array | ((-0.5 -1 10)) | array | ((2.5 -1 10)) | array | ((4.5 -1 10)) | array | ((6.5 -1 10)) | array | ((8 -1 10)) | array | ((-10 -1 12)) | array | ((-9.5 -1 12)) | array | ((-8.5 -1 12)) | array | ((-0.5 -1 12)) | array | ((2.5 -1 12)) | array | ((3 -1 12)) | array | ((6.5 -1 12)) | array | ((9.5 -1 12)) | array | ((-12 -1 14)) | array | ((-11 -1 14)) | array | ((-10 -1 14)) | array | ((-9.5 -1 14)) | array | ((-8.5 -1 14)) | array | ((-0.5 -1 14)) | array | ((9 -1 14)) | array | ((11 -1 14)) | array | ((-13 -1 16)) | array | ((-12.5 -1 16)) | array | ((-7.5 -1 16)) | array | ((2.5 -1 16)) | array | ((-13 -1 18)) | array | ((-8 -1 18)) | array | ((-0.5 -1 18)) | array | ((5.5 -1 18)) | array | ((10 -1 18)) | array | ((10.5 -1 18)) | array | ((14 -1 18)) | array | ((-14.5 -1 20)) | array | ((-14 -1 20)) | array | ((3 -1 20)) | array | ((9 -1 20)) | array | ((9.5 -1 20)) | array | ((10.5 -1 20)) | array | ((11 -1 20)) | array | ((13 -1 20)) | array | ((13.5 -1 20)) | array | ((-13 -1 22)) | array | ((-11 -1 22)) | array | ((-10.5 -1 22)) | array | ((11.5 -1 22)) | array | ((14 -1 22)) | array | ((14.5 -1 22)) | array | ((-14.5 -1 24)) | array | ((-11.5 -1 24)) | array | ((3 -1 24)) | array | ((12.5 -1 24)) | array | ((14 -1 24)) | array | ((-12.5 -1 26)) | array | ((-2.5 -1 26)) | array | ((3 -1 26)) | array | ((4 -1 26)) | array | ((13.5 -1 26)) | array | ((-15 -1 28)) | array | ((-13.5 -1 28)) | array | ((-2.5 -1 28)) | array | ((13 -1 28)) | array | ((14.5 -1 28)) | array | ((-14.5 -1 30)) | array | ((-15 -1 32)) | array | ((3.5 -1 38)) |

#   As this video is in "extended mode", you can see both the animated
#   grid windows, and the evolving heat map.  Note that here, we
#   actually have moved away from ~image_plane_scan~ for generating the
#   grid windows, and in fact are using ~random_scan4~ which, the reader
#   will recall, lays out grids randomly in "physical space" in a volume
#   that corresponds to a long, thick "ribbon".  Here, the ribbon
#   extends across the road and from the road up to camera height.  The
#   randomness of the windows should be self-evident from the video.

#   There still are a few transient false-positives at a few places,
#   unfortunately.  More tuning of the cooling, thresholding, and other
#   parameters might banish them.  

#   #+HTML: <iframe width="800" height="450" src="https://www.youtube.com/embed/yGHN0OlBVRU" frameborder="0" allowfullscreen></iframe>

#   Finally, we repeat processing for the main project video, but not in
#   extended mode.  This mainly was so that I could get a sense of the
#   difference in processing times.  In fact, it cuts the processing
#   time by about a half, indicating that generating, resizing, and
#   composing together all the windows introduces considerable
#   overhead.

in_clip = VideoFileClip("project_video.mp4")
scene = Component(scale(mpimg.imread("test_images/test1.jpg")),threshold=40,cooling_factor=0.99,extended=False)
out_clip = in_clip.fl_image(scene.process_image)
out_clip.write_videofile("output_images/project_output.mp4", audio=False)
