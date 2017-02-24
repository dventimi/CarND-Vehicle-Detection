from glob import glob
from itertools import groupby, islice, zip_longest, cycle, filterfalse, chain
from lesson_functions import *
from moviepy.editor import VideoFileClip, VideoClip
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from random import choice, sample
from scipy.cluster.vq import kmeans,vq
from scipy.ndimage.measurements import label
from scipy.stats import gaussian_kde
from skimage import color, exposure
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import builtins
import cProfile
import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import timeit

plt.ion()

feed = lambda pattern, y: ((f, y) for f in glob(pattern))
shuffle = lambda l: sample(l, len(l))
scale = lambda img: (img/np.max(img)*255).astype(np.uint8)
load = lambda g: ((mpimg.imread(x[0]),x[1]) for x in g)
flip = lambda g: ((x[0][:,::-1,:],x[1]) for x in g)
mirror = lambda g: chain(g, flip(g))

class HyperParameters:
    def __init__(self):
        pass


global Theta
Theta = HyperParameters()
Theta.cell_per_block = 4
Theta.channel = 2
Theta.colorspace = cv2.COLOR_RGB2HSV
Theta.feature_vec = True
Theta.orient = 15
Theta.pix_per_cell = 16
Theta.transform_sqrt = False
Theta.horizon = 0.28
Theta.height = 2
Theta.width = 2
Theta.spatial_size = 32


def extract_features(img):
    img = scale(img)
    X = np.array([])
    X = np.append(X,
                  hog(cv2.cvtColor(img, Theta.colorspace)[:,:,2],
                      Theta.orient,
                      (Theta.pix_per_cell,Theta.pix_per_cell),
                      (Theta.cell_per_block,Theta.cell_per_block),
                      transform_sqrt = Theta.transform_sqrt,
                      feature_vector = Theta.feature_vec))
    s = StandardScaler().fit(X)
    return s.transform(X)


def get_classifier(X, y):
    svc = LinearSVC()
    svc.fit(X, y)
    return svc


samples = list(chain(feed("vehicles/**/*.png",1),feed("non-vehicles/**/*.png",0)))
data = cycle(mirror(load(shuffle(samples))))

X_train,X_test,y_train,y_test = train_test_split(*zip(*((extract_features(s[0]), s[1]) for s in islice(data, len(samples)))), test_size=0.2, random_state=np.random.randint(0, 100))
classifier = get_classifier(X_train,y_train)
print('Test Accuracy of SVC = ', round(classifier.score(X_test, y_test), 4))

def draw_window(img, bbox, color=(0,0,255), thick=3):
    cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    return img


def get_patches(img, grid, size=(64,64)):
    return ((cv2.resize(img[window[0][1]:window[1][1],
                            window[0][0]:window[1][0]],size),window) for window in grid)


clip_window = lambda x, box: sum([box[0]<=x[0][0]<=box[1],
                                  box[0]<=x[1][0]<=box[1],
                                  box[2]<=x[0][1]<=box[3],
                                  box[2]<=x[1][1]<=box[3]])==4


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
    grid = np.mgrid[0:img.shape[1]:delta,img.shape[0]:-delta:-delta].T.reshape(-1,2)
    grid = ([(c[0],c[1]), (c[0]+size,c[1]+size)] for c in grid)
    grid = filter(lambda x: clip_window(x, box1), grid)
    grid = filter(lambda x: clip_window(x, box2), grid)
    return grid
    
image = scale(mpimg.imread("test_images/test1.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]),
                   chain(
                       image_plane_scan(image,4,0.75,1),
                       image_plane_scan(image,4,0.75,2),
                       image_plane_scan(image,4,0.75,3)
                   )))))
mpimg.imsave("output_images/imageplane-scan.png", image, format="png")


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

image = scale(mpimg.imread("test_images/test1.jpg"))
grid = chain(
    image_plane_scan(image,4,0.50,1),
    image_plane_scan(image,4,0.50,2),
    image_plane_scan(image,4,0.50,3),
)
grid = list(grid)

def process(x):
    return (classifier.predict(extract_features(x[0]))[0],x[1])


try:
    pool = Pool(12)
    builtins.__dict__.update(locals())
    results = pool.map(process, get_patches(image, grid))
finally:
    pool.close()
    pool.join()

box_list = list(map(lambda x: x[1][:2], filter(lambda x: x[0]>0, results)))
heat = np.zeros_like(image[:,:,0]).astype(np.float)
heat = add_heat(heat,box_list)
heat = apply_threshold(heat,10)
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

def get_processor(pool, grid):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    def process_image(img):
        image = np.copy(img)
        results = pool.map(process, get_patches(image, grid))
        box_list = list(map(lambda x: x[1][:2], filter(lambda x: x[0]>0, results)))
        add_heat(heat,box_list)
        return scale(np.dstack((heat,heat,heat)))
    return process_image


builtins.__dict__.update(locals())
in_clip = VideoFileClip("project_video.mp4")
try:
    pool = Pool(8)
    out_clip = in_clip.fl_image(get_processor(pool, grid))
    out_clip.write_videofile("output_images/project_output.mp4", audio=False)
finally:
    pool.close()
    pool.join()

