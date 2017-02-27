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
Theta.orient = 9
Theta.pix_per_cell = 8
Theta.transform_sqrt = False
Theta.test_size = 0.2

def extract_features(img):
    img = scale(img)
    X = np.array([])
    X = np.append(X,
                  hog(cv2.cvtColor(img, Theta.colorspace)[:,:,Theta.channel],
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

X_train,X_test,y_train,y_test = train_test_split(*zip(*((extract_features(s[0]), s[1]) for s in islice(data, len(samples)))), test_size=Theta.test_size, random_state=np.random.randint(0, 100))
classifier = get_classifier(X_train,y_train)
print('Test Accuracy of SVC = ', round(classifier.score(X_test, y_test), 4))

def draw_window(img, bbox, color=(0,0,255), thick=3):
    cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    return img


def get_patches(img, grid, size=(64,64)):
    return ((cv2.resize(img[window[0][1]:window[1][1],
                            window[0][0]:window[1][0]],size),window) for window in grid)


def clip_window(x, box):
    return sum([box[0]<=x[0][0]<=box[1],
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
    grid = np.mgrid[0:img.shape[1]:delta,
                    img.shape[0]:-delta:-delta].T.reshape(-1,2)
    grid = ([(c[0],c[1]), (c[0]+size,c[1]+size)] for c in grid)
    grid = filter(lambda x: clip_window(x, box1), grid)
    grid = filter(lambda x: clip_window(x, box2), grid)
    return grid
    

image = scale(mpimg.imread("test_images/test1.jpg"))
print("Number of windows: %s" %
      len(list(map(lambda w: draw_window(image, w[:2]),
                   chain(
                       image_plane_scan(image,4,0.50,1),
                       image_plane_scan(image,4,0.50,2),
                       image_plane_scan(image,4,0.50,3)
                   )))))
mpimg.imsave("output_images/imageplane-scan.png", image, format="png")


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1],
                box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heat, threshold):
    heatmap = np.copy(heat)
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        center = (np.mean((np.min(nonzerox), np.max(nonzerox))),
                  np.mean((np.min(nonzeroy), np.max(nonzeroy))))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        cv2.putText(img, "Car: %s" % car_number,
                    (bbox[0][0],bbox[0][1]-20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img, "Center: %s" % (center,),
                    (bbox[0][0],bbox[0][1]-10),
                    cv2.FONT_HERSHEY_DUPLEX, .5, (255,255,255), 1)
    return img


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


image = scale(mpimg.imread("test_images/test1.jpg"))
print("Number of windows: %s" %
      len(list(map(lambda w: draw_window(image, w[:2]),
                   chain(
                       random_scan(image,180),
                       random_scan(image,90),
                       random_scan(image,60)
                   )))))
mpimg.imsave("output_images/random-scan1.png", image, format="png")


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


image = scale(mpimg.imread("test_images/test1.jpg"))
print("Number of windows: %s" %
      len(list(map(lambda w: draw_window(image, w[:2]),
                   chain(
                       random_scan2(image,256,1000),
                       random_scan2(image,128,1000),
                       random_scan2(image,64,1000)
                   )))))
mpimg.imsave("output_images/random-scan2.png", image, format="png")


def random_scan3(img,size,num=100):
    p = np.random.rand(num,2)
    p[:,0]*=image.shape[1]
    p[:,1]*=math.pi*2
    p = p[p[:,0]>image.shape[0]//3]
    p = p[p[:,1]>0]
    p = p[p[:,1]<math.pi]
    s = (size//2*p[:,0]/image.shape[1]).astype('int')
    x,y=zip(*np.dstack((image.shape[1]//2+p[:,0]*np.cos(p[:,1]), image.shape[0]//2+p[:,0]*np.sin(p[:,1]))).astype('int')[0])
    grid = ([(c[0]-c[2],c[1]-c[2]), (c[0]+c[2],c[1]+c[2])] for c in zip(x,y,s))
    box = (0,img.shape[1],(img.shape[0]//2),670)
    grid = filter(lambda x: clip_window(x, box), grid)
    return grid
    

image = scale(mpimg.imread("test_images/test1.jpg"))
print("Number of windows: %s" %
      len(list(map(lambda w: draw_window(image, w[:2]),
                   chain(
                       random_scan3(image,image.shape[1]//4,3000)
                   )))))
mpimg.imsave("output_images/random-scan3.png", image, format="png")

def process(x):
    return (classifier.predict(extract_features(x[0]))[0],x[1])


image = scale(mpimg.imread("test_images/test1.jpg"))
grid = list(chain(
    image_plane_scan(image,4,0.750,1),
    image_plane_scan(image,4,0.750,2),
    image_plane_scan(image,4,0.750,3),
))

builtins.__dict__.update(locals())
try:
    pool = Pool(12)
    results = pool.map(process, get_patches(image, grid))
finally:
    pool.close()
    pool.join()

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




class Component:
    def overlap(c1, c2):
        distance = ((c1.center()[0]-c2.center()[0])**2 +
                    (c1.center()[1]-c2.center()[1])**2)**0.5
        return distance < c1.size() + c2.size()


    def __init__(self, pool, img, origin=None, size=None):
        self.pool = pool
        self.image = img
        self.mainwindow = np.copy(image)
        self.bboxwindow = np.copy(image)
        self.heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        self.flat = np.zeros_like(image[:,:,0]).astype(np.float)
        self.labels = None
        self.children = []
        self.origin = origin if origin else tuple(np.array(img.shape[:2][::-1])//2)
        self.size = size if size else min(img.shape[:2])//2


    def center(self):
        return self.center


    def size(self):
        return self.size


    def cool(self):
        self.heatmap*=0.98


    def get_heatmap(self):
        return self.heatmap


    def sample(self, pool, mainwindow, grid):
        results = pool.map(process, get_patches(mainwindow, grid))
        return results


    def heat(self, results):
        samples = list(map(lambda x: x[1][:2], filter(lambda x: x[0]>0, results)))
        for s in samples:
            self.heatmap[s[0][1]:s[1][1],
                         s[0][0]:s[1][0]] += 1
        # list(map(lambda x: heatmap[s[0][1]:s[1][1], s[0][0]:s[1][0]] += 1, samples))


    def evolve(self, image):
        self.cool()
        self.mainwindow = np.copy(image)
        self.bboxwindow = np.copy(image)
        grid = self.grid(1000)
        self.addboxes(self.bboxwindow, grid)
        results = self.sample(self.pool, self.mainwindow, grid)
        self.heat(results)
        thresholded = apply_threshold(self.get_heatmap(),20)
        self.labels = label(thresholded)
        draw_labeled_bboxes(self.mainwindow, labels)


    def get_out_img(self):
        bbox_img = cv2.resize(self.bboxwindow, tuple(np.array(self.image.shape[:2][::-1])//2))
        heat_img = cv2.resize(np.dstack([self.get_heatmap(), self.get_heatmap(), self.flat]),
                              tuple(np.array(image.shape[:2][::-1])//2))
        outp_img = cv2.resize(np.hstack((np.vstack((self.mainwindow, np.hstack((bbox_img,heat_img)))), np.vstack((heat_img, heat_img, heat_img)))), tuple(np.array(self.image.shape[:2][::-1])))
        cv2.putText(outp_img, "Max: %.2f" % np.max(self.get_heatmap()), (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
        cv2.putText(outp_img, "Cars: %s" % self.labels[1], (50,80), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
        return outp_img


    def grid(self, num):
        return list(random_scan3(self.image, self.image.shape[1]//4, num))


    def addboxes(self, bboxwindow, grid):
        list(map(lambda w: draw_window(bboxwindow, w[:2]), grid))


    def spawn(self):
        thresholded = apply_threshold(self.heatmap,1)
        labels = label(thresholded)
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            size = min(bbox[0][1]-bbox[0][0], bbox[1][1]-bbox[1][0])
            center = (np.mean((np.min(nonzerox), np.max(nonzerox))),
                      np.mean((np.min(nonzeroy), np.max(nonzeroy))))
            center = center


    def process_image(self, image):
        self.evolve(image)
        return self.get_out_img()


builtins.__dict__.update(locals())
in_clip = VideoFileClip("test_video.mp4")
try:
    pool = Pool(8)
    scene = Component(pool, scale(mpimg.imread("test_images/test1.jpg")))
    out_clip = in_clip.fl_image(scene.process_image)
    out_clip.write_videofile("output_images/test_output.mp4", audio=False)
finally:
    pool.close()
    pool.join()
