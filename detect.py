from glob import glob
from itertools import groupby, islice, zip_longest, cycle, filterfalse, chain
from moviepy.editor import VideoFileClip, VideoClip
from random import choice, sample
from skimage import color, exposure
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


crt2cyl = lambda x,y,z: (math.sqrt(x**2+y**2), math.atan2(y,x), z)
cyl2crt = lambda rho,phi,z: (rho*math.cos(phi), rho*math.sin(phi), z)
cyl2sph = lambda rho,phi,z: (math.sqrt(rho**2+z**2), math.atan2(rho, z), phi)
sph2cyl = lambda r,theta,phi: (r*math.sin(theta), phi, r*math.cos(theta))
crt2sph = lambda x,y,z: (math.sqrt(x**2+y**2+z**2), math.acos(z/math.sqrt(x**2+y**2+z**2)), math.atan2(y,x))
sph2crt = lambda r,theta,phi: (r*math.sin(theta)*math.cos(phi), r*math.sin(theta)*math.sin(phi), r*math.cos(theta))


feed = lambda pattern, y: ((f, y) for f in glob(pattern))
shuffle = lambda l: sample(l, len(l))
load = lambda g: ((mpimg.imread(x[0]),x[1]) for x in g)
flip = lambda g: ((x[0][:,::-1,:],x[1]) for x in g)
mirror = lambda g: chain(g, flip(g))
group = lambda items, n, fillvalue=None: zip_longest(*([iter(items)]*n), fillvalue=fillvalue)
transpose = lambda tuples: (list(map(list, zip(*g))) for g in tuples)
batch = lambda groups, indices=[0, 1]: ([np.asarray(t[i]) for i in indices] for t in groups)


def get_patch(img, x, y, z, horizon=0.5, width=4, height=4):
    d = 1
    r,theta,phi = crt2sph(x,y,z)
    rho2 = d*math.tan(theta)
    x2,y2 = (rho2*math.cos(phi),rho2*math.sin(phi))
    center = (int(img.shape[1]*0.5+x2*img.shape[1]//2),
              int(img.shape[0]*(1-horizon)-y2*img.shape[1]//2))
    scale = img.shape[1]//2
    dx = int(width/2*scale/z)
    dy = int(height/2*scale/z)
    bbox = [(center[0]-dx,center[1]-dy), (center[0]+dx,center[1]+dy)]
    return (bbox,center)


def draw_patch(img, bbox, color=(0,0,255), thick=3):
    cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    return img


def get_frame_maker(img):
    x = -15
    y = - 1.5
    def make_frame(t):
        frame = np.copy(img)
        z = 2**(t % 5)*5
        draw_patch(frame, get_patch(frame,-15,y,z,horizon=0.30)[1], color=(0,255,255))
        draw_patch(frame, get_patch(frame,-7,y,z,horizon=0.30)[1], color=(0,255,255))
        draw_patch(frame, get_patch(frame,0,y,z,horizon=0.30)[1], color=(0,255,255))
        draw_patch(frame, get_patch(frame,7,y,z,horizon=0.30)[1], color=(0,255,255))
        draw_patch(frame, get_patch(frame,15,y,z,horizon=0.30)[1], color=(0,255,255))
        return frame
    return make_frame


clip = VideoClip(get_frame_maker(mpimg.imread('bbox-example-image.jpg')), duration=5)
clip.write_videofile("output_images/patches1.mp4", fps=25)


class HyperParameters:
    def __init__(self):
        pass


global Theta
Theta = HyperParameters()
Theta.colorspace = cv2.COLOR_RGB2HSV
Theta.channel = 1
Theta.orient = 9
Theta.pix_per_cell = 8
Theta.cell_per_block = 2
Theta.transform_sqrt = False
Theta.feature_vec = True

def extract_features(img):
    X = np.array([])
    X = np.append(X, hog(cv2.cvtColor(img, Theta.colorspace)[:,:,0],
                         Theta.orient,
                         (Theta.pix_per_cell,Theta.pix_per_cell),
                         (Theta.cell_per_block,Theta.cell_per_block),
                         transform_sqrt = Theta.transform_sqrt,
                         feature_vector = Theta.feature_vec))
    X = np.append(X, hog(cv2.cvtColor(img, Theta.colorspace)[:,:,1],
                         Theta.orient,
                         (Theta.pix_per_cell,Theta.pix_per_cell),
                         (Theta.cell_per_block,Theta.cell_per_block),
                         transform_sqrt = Theta.transform_sqrt,
                         feature_vector = Theta.feature_vec))
    X = np.append(X, hog(cv2.cvtColor(img, Theta.colorspace)[:,:,2],
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

Theta.colorspace = cv2.COLOR_RGB2HSV
Theta.channel = 2
Theta.orient = 15
Theta.pix_per_cell = 16
Theta.cell_per_block = 4
Theta.transform_sqrt = False
Theta.feature_vec = True

X_train,X_test,y_train,y_test = train_test_split(*zip(*((extract_features(s[0]), s[1]) for s in islice(data, len(samples)))), test_size=0.2, random_state=np.random.randint(0, 100))
classifier = get_classifier(X_train,y_train)
print('Test Accuracy of SVC = ', round(classifier.score(X_test, y_test), 4))


def get_processor(image):
    grid = [get_patch(image, *x) for x in np.mgrid[-2:-0.5:0.1,
                                                   -16:16:1,
                                                   10:100:10].T.reshape(-1,3)]
    def process_image(img):
        frame = np.copy(img)
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        for window,center in grid:
            import pdb
            pdb.set_trace()
            test_img = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = extract_features(test_img)
            prediction = clf.predict(test_features)
            if prediction==1:
                np.zeros[center]+=1
        return heat
    return process_image


in_clip = VideoFileClip("project_video.mp4")
out_clip = in_clip.fl_image(get_processor(5))
cProfile.run('out_clip.write_videofile("output_images/project_output.mp4", audio=False)', 'restats')



        
