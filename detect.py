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
import pickle
import time
import timeit

# plt.ion()

feed = lambda pattern, y: ((f, y) for f in glob(pattern))
shuffle = lambda l: sample(l, len(l))
scale = lambda img,maxval=None: (img/np.max(img)*255).astype(np.uint8) if maxval==None else (img/maxval*255).astype(np.uint8)
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
Theta.threshold = 25
Theta.numwindows = 100
Theta.cooling_factor = 0.98

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


def get_classifier(reload=False):
    if reload:
        samples = list(chain(feed("vehicles/**/*.png",1),feed("non-vehicles/**/*.png",0)))
        data = cycle(mirror(load(shuffle(samples))))
        X_train,X_test,y_train,y_test = train_test_split(*zip(*((extract_features(s[0]), s[1]) for s in islice(data, len(samples)))), test_size=Theta.test_size, random_state=np.random.randint(0, 100))
        svc = LinearSVC()
        svc.fit(X, y)
        pickle.dump(svc, open("save.p","wb"))
        print('Test Accuracy of SVC = ', round(classifier.score(X_test, y_test), 4))
    else:
        svc = pickle.load(open("save.p", "rb"))
    return svc

classifier = get_classifier()

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


clip_window2 = lambda x, img: sum([0<=x[0][0]<=image.shape[1],
                                   0<=x[1][0]<=image.shape[1],
                                   0<=x[0][1]<=image.shape[0],
                                   0<=x[1][1]<=image.shape[0]])==4



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



crt2cyl = lambda x,y,z: (math.sqrt(x**2+y**2), math.atan2(y,x), z)
cyl2crt = lambda rho,phi,z: (rho*math.cos(phi), rho*math.sin(phi), z)
cyl2sph = lambda rho,phi,z: (math.sqrt(rho**2+z**2), math.atan2(rho, z), phi)
sph2cyl = lambda r,theta,phi: (r*math.sin(theta), phi, r*math.cos(theta))
crt2sph = lambda x,y,z: (math.sqrt(x**2+y**2+z**2), math.acos(z/math.sqrt(x**2+y**2+z**2)), math.atan2(y,x))
sph2crt = lambda r,theta,phi: (r*math.sin(theta)*math.cos(phi), r*math.sin(theta)*math.sin(phi), r*math.cos(theta))


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
    grid = (get_window(img,x[0],x[1],x[2])+[x] for x in grid)
    grid = filter(lambda x: clip_window2(x, img), grid)
    return grid

image = scale(mpimg.imread("test_images/test1.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), sparse_scan(image)))))
mpimg.imsave("output_images/sparse-scan.png", image, format="png")

image = scale(mpimg.imread("test_images/test1.jpg"))
clip = VideoClip(get_frame_maker(image, cycle(sparse_scan(image))), duration=10)
clip.write_videofile("output_images/sparse-scan.mp4", fps=25)

def dense_scan(img, h=2,w=2):
    grid = np.mgrid[-12:14:0.5,-1.0:0:2,5:50:2]
    grid = grid.T.reshape(-1,3)
    grid = (get_window(img,x[0],x[1],x[2], height=h, width=w)+[x] for x in grid)
    grid = filter(lambda x: clip_window2(x, img), grid)
    return grid


image = scale(mpimg.imread("test_images/test1.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), dense_scan(image)))))
mpimg.imsave("output_images/dense-scan.png", image, format="png")

image = scale(mpimg.imread("test_images/test1.jpg"))
clip = VideoClip(get_frame_maker(image, cycle(dense_scan(image))), duration=20)
clip.write_videofile("output_images/dense-scan.mp4", fps=60)

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
    grid = filter(lambda x: clip_window2(x, img), grid)
    return grid


image = scale(mpimg.imread("test_images/test1.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), random_scan4(image,2,1000)))))
mpimg.imsave("output_images/random-scan4.png", image, format="png")

image = scale(mpimg.imread("test_images/test1.jpg"))
clip = VideoClip(get_frame_maker(image, cycle(random_scan4(image,2,1000))), duration=20)
clip.write_videofile("output_images/random-scan4.mp4", fps=60)


def random_scan5(img,size,num=100):
    grid = chain(random_scan4(img,size,num//2,width=20,left=-30),
                 random_scan4(img,size,num//2,width=20,left=+10))
    return grid


image = scale(mpimg.imread("test_images/test1.jpg"))
print(len(list(map(lambda w: draw_window(image, w[:2]), random_scan5(image,2,1000)))))
mpimg.imsave("output_images/random-scan5.png", image, format="png")

image = scale(mpimg.imread("test_images/test1.jpg"))
clip = VideoClip(get_frame_maker(image, cycle(random_scan5(image,2,1000))), duration=20)
clip.write_videofile("output_images/random-scan5.mp4", fps=60)



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
    def __init__(self, pool, img, center=None, size=None):
        self.pool = pool
        self.image = img
        self.mainwindow = np.copy(image)
        self.bboxwindow = np.copy(image)
        self.heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        self.flat = np.zeros_like(image[:,:,0]).astype(np.float)
        self.labels = None
        self.children = []
        self.center = center if center else tuple(np.array(img.shape[:2][::-1])//2)
        self.size = size if size else min(img.shape[:2])//2


    def get_center(self):
        return self.center


    def get_size(self):
        return self.size


    def cool(self):
        self.heatmap*=Theta.cooling_factor


    def get_heatmap(self):
        return self.heatmap


    def sample(self, mainwindow, grid):
        # results = self.pool.map(process, get_patches(mainwindow, grid))
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
        grid = self.grid(Theta.numwindows)
        self.addboxes(self.bboxwindow, grid)
        results = self.sample(self.mainwindow, grid)
        self.heat(results)
        self.heatmap = cv2.GaussianBlur(self.heatmap, (31, 31), 0)
        thresholded = apply_threshold(self.get_heatmap(),Theta.threshold)
        self.labels = label(thresholded)
        draw_labeled_bboxes(self.mainwindow, self.labels)


    def get_out_img(self):
        bbox_img = cv2.resize(self.bboxwindow, tuple(np.array(self.image.shape[:2][::-1])//2))
        hot2_img = cv2.resize(scale(np.dstack([self.get_heatmap(), self.get_heatmap(), self.flat]), 2*Theta.threshold), tuple(np.array(image.shape[:2][::-1])//2))
        cv2.putText(hot2_img, "Max: %.2f" % np.max(self.get_heatmap()), (25,25), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
        flat_img = cv2.resize(np.dstack([self.flat, self.flat, self.flat]), tuple(np.array(image.shape[:2][::-1])//2))
        outp_img = cv2.resize(np.hstack((np.vstack((self.mainwindow,
                                                    np.hstack((flat_img,
                                                               flat_img)))),
                                         np.vstack((bbox_img,
                                                    hot2_img,
                                                    flat_img)))),
                              tuple(np.array(self.image.shape[:2][::-1])))
        return outp_img


    def grid(self, num):
        return list(random_scan5(self.image,
                                 2,
                                 num))


    def addboxes(self, bboxwindow, grid):
        list(map(lambda w: draw_window(bboxwindow, w[:2]), grid))


    def process_image(self, image):
        self.evolve(image)
        return self.get_out_img()


builtins.__dict__.update(locals())
try:
    in_clip = VideoFileClip("project_video.mp4")
    pool = Pool(8)
    scene = Component(pool, scale(mpimg.imread("test_images/test1.jpg")))
    out_clip = in_clip.fl_image(scene.process_image)
    # out_clip.write_videofile("output_images/test_output.mp4", audio=False)
    cProfile.run('out_clip.write_videofile("output_images/project_output.mp4", audio=False)', 'restats')
finally:
    pool.close()
    pool.join()
    in_clip = None


# import pstats
# p = pstats.Stats('restats')
# p.sort_stats('time').print_stats(100)
# p.sort_stats('cumulative').print_stats(100)
