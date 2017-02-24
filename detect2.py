from glob import glob
from itertools import groupby, islice, zip_longest, cycle, filterfalse, chain
from moviepy.editor import VideoFileClip, VideoClip
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from random import choice, sample
from skimage import color, exposure
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import builtins
import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
import timeit

plt.ion()

crt2cyl = lambda x,y,z: (math.sqrt(x**2+y**2), math.atan2(y,x), z)
cyl2crt = lambda rho,phi,z: (rho*math.cos(phi), rho*math.sin(phi), z)
cyl2sph = lambda rho,phi,z: (math.sqrt(rho**2+z**2), math.atan2(rho, z), phi)
sph2cyl = lambda r,theta,phi: (r*math.sin(theta), phi, r*math.cos(theta))
crt2sph = lambda x,y,z: (math.sqrt(x**2+y**2+z**2), math.acos(z/math.sqrt(x**2+y**2+z**2)), math.atan2(y,x))
sph2crt = lambda r,theta,phi: (r*math.sin(theta)*math.cos(phi), r*math.sin(theta)*math.sin(phi), r*math.cos(theta))


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



def extract_features(img):
    img = scale(img)
    X = np.array([])
    X = np.append(X, hog(cv2.cvtColor(img, Theta.colorspace)[:,:,Theta.channel],
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


def draw_window(img, bbox, color=(0,0,255), thick=3):
    cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    return img


def get_patches(img, grid, size=(64,64)):
    return ((cv2.resize(image[window[0][1]:window[1][1],
                              window[0][0]:window[1][0]],size),window[2]) for window in grid)


image = scale(mpimg.imread("bbox-example-image.jpg"))
draw_window(image, get_window(image, 0, 0.0, 1, horizon=0.5, width=2, height=1))
mpimg.imsave("output_images/windshield.png", image, format="png")
plt.imshow(image)

image = scale(mpimg.imread("bbox-example-image.jpg"))
draw_window(image, get_window(image, 4.1, -1.0, 8, horizon=0.28))
draw_window(image, get_window(image, -10.5, -1.0, 22, horizon=0.28))
draw_window(image, get_window(image, -6.1, -1.0, 32, horizon=0.28))
draw_window(image, get_window(image, -0.8, -1.0, 35, horizon=0.28))
draw_window(image, get_window(image, 3, -1.0, 55, horizon=0.28))
draw_window(image, get_window(image, -6.1, -1.0, 55, horizon=0.28))
draw_window(image, get_window(image, -6.1, -1.0, 70, horizon=0.28))
mpimg.imsave("output_images/bbox-example-image-test.png", image, format="png")
plt.imshow(image)

clip_window = lambda x, img: sum([0<=x[0][0]<=image.shape[1],
                                  0<=x[1][0]<=image.shape[1],
                                  0<=x[0][1]<=image.shape[0],
                                  0<=x[1][1]<=image.shape[0]])==4

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
clip = VideoClip(zooming_windows(mpimg.imread('bbox-example-image.jpg')), duration=5)
clip.write_videofile("output_images/zooming-windows.mp4", fps=25)

def get_frame_maker(img, grid):
    def make_frame(t):
        frame = np.copy(img)
        draw_window(frame, grid.__next__()[:2], color=(0,255,255))
        return frame
    return make_frame

def sparse_scan(img):
    grid = np.mgrid[-10:12:2,-1.0:0:2,3:7:1]
    grid[2,]=2**grid[2,]
    grid = grid.T.reshape(-1,3)
    grid = (get_window(img,x[0],x[1],x[2], horizon=0.28)+[x] for x in grid)
    grid = filter(lambda x: clip_window(x, img), grid)
    return grid

image = scale(mpimg.imread("bbox-example-image.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), sparse_scan(image)))))
mpimg.imsave("output_images/sparse-scan.png", image, format="png")

image = scale(mpimg.imread("bbox-example-image.jpg"))
clip = VideoClip(get_frame_maker(image, cycle(sparse_scan(image))), duration=10)
clip.write_videofile("output_images/sparse-scan.mp4", fps=25)

def dense_scan(img, h=2,w=2):
    grid = np.mgrid[-12:14:0.5,-1.0:0:2,5:50:2]
    grid = grid.T.reshape(-1,3)
    grid = (get_window(img,x[0],x[1],x[2], horizon=0.28, height=h, width=w)+[x] for x in grid)
    grid = filter(lambda x: clip_window(x, img), grid)
    return grid

image = scale(mpimg.imread("bbox-example-image.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), dense_scan(image)))))
mpimg.imsave("output_images/dense-scan.png", image, format="png")

image = scale(mpimg.imread("bbox-example-image.jpg"))
clip = VideoClip(get_frame_maker(image, cycle(dense_scan(image))), duration=20)
clip.write_videofile("output_images/dense-scan.mp4", fps=60)

def perimeter_scan(img):
    grid = np.mgrid[-12:14:0.5,-1.0:0:2,5:50:2]
    grid = grid.T.reshape(-1,3)
    grid = list(filter(lambda x: not (-4<=x[0]<=4 and 5<=x[2]<=40), grid))
    grid = (get_window(img,x[0],x[1],x[2], horizon=0.28)+[x] for x in grid)
    grid = filter(lambda x: clip_window(x, img), grid)
    return grid

image = scale(mpimg.imread("bbox-example-image.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), perimeter_scan(image)))))
mpimg.imsave("output_images/perimeter-scan.png", image, format="png")

image = scale(mpimg.imread("bbox-example-image.jpg"))
clip = VideoClip(get_frame_maker(image, cycle(perimeter_scan(image))), duration=20)
clip.write_videofile("output_images/perimeter-scan.mp4", fps=60)

def local_scan(img, nodes):
    grid = chain((np.mgrid[n[0]-1:n[0]+2:0.5,
                           -1.0:0:2,
                           n[2]-1:n[2]+2:0.5].T.reshape(-1,3)) for n in nodes)
    grid = np.concatenate(list(grid))
    grid = (get_window(img,x[0],x[1],x[2], horizon=0.28)+[x] for x in grid)
    grid = filter(lambda x: clip_window(x, img), grid)
    return grid

nodes = [(4.1, -1.0, 8),
         (-10.5, -1.0, 22),
         (-6.1, -1.0, 32),
         (-0.8, -1.0, 35),
         (3, -1.0, 55),
         (-6.1, -1.0, 55),
         (-6.1, -1.0, 70)]

image = scale(mpimg.imread("bbox-example-image.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), local_scan(image, nodes)))))
mpimg.imsave("output_images/local-scan.png", image, format="png")

image = scale(mpimg.imread("bbox-example-image.jpg"))
clip = VideoClip(get_frame_maker(image, cycle(local_scan(image, nodes))), duration=10)
clip.write_videofile("output_images/local-scan.mp4", fps=25)

def process(x):
    return (classifier.predict(extract_features(x[0]))[0],x[1])


builtins.__dict__.update(locals())
try:
    pool = Pool(12)
    results = pool.map(process, get_patches(image, dense_scan(image)))
finally:
    pool.close()
    pool.join

_,r = zip(*filter(lambda x: x[0]>0, results))

x,y,z = zip(*r)

plt.scatter(x,z,s=10,c=y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
