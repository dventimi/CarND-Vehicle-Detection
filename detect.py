from itertools import groupby, islice, zip_longest, cycle, filterfalse, chain
from moviepy.editor import VideoFileClip, VideoClip
from skimage import color, exposure
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import cv2
import glob
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


class HyperParameters:
    def __init__(self):
        pass


global Theta
Theta = HyperParameters()
Theta.colorspace = 'RGB'
Theta.orient = 9
Theta.pix_per_cell = 8
Theta.cell_per_block = 2
Theta.hog_channel = 0

def extract_features(img):
    X = np.array([])
    X = np.append(X, hog(cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,0]))
    s = StandardScaler().fit(X)
    return s.transform(X)


X,y = zip(*chain(((extract_features(mpimg.imread(f)), 1) for f in glob.glob('vehicles_smallset/**/*.jpeg')),
                 ((extract_features(mpimg.imread(f)), 0) for f in glob.glob('non-vehicles_smallset/**/*.jpeg'))))

def get_classifier(X, y):
    svc = LinearSVC()
    svc.fit(X, y)
    return svc


c = get_classifier(X,y)


crt2cyl = lambda x,y,z: (math.sqrt(x**2+y**2), math.atan2(y,x), z)
cyl2crt = lambda rho,phi,z: (rho*math.cos(phi), rho*math.sin(phi), z)
cyl2sph = lambda rho,phi,z: (math.sqrt(rho**2+z**2), math.atan2(rho, z), phi)
sph2cyl = lambda r,theta,phi: (r*math.sin(theta), phi, r*math.cos(theta))
crt2sph = lambda x,y,z: (math.sqrt(x**2+y**2+z**2), math.acos(z/math.sqrt(x**2+y**2+z**2)), math.atan2(y,x))
sph2crt = lambda r,theta,phi: (r*math.sin(theta)*math.cos(phi), r*math.sin(theta)*math.sin(phi), r*math.cos(theta))


def draw_patch(img, x, y, z, horizon=0.5, width=4, height=4, color=(0,0,255), thick=6):
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
    cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    return img


image = mpimg.imread('bbox-example-image.jpg')
for d in 2**np.arange(6)*5:
    plt.imshow(draw_patch(image, -15, -1.5, d, horizon=0.3, color=(0,255,255))) 
    plt.imshow(draw_patch(image, -7,  -1.5, d, horizon=0.3, color=(255,0,0)))
    plt.imshow(draw_patch(image,  0,  -1.5, d, horizon=0.3, color=(255,255,255)))
    plt.imshow(draw_patch(image,  7,  -1.5, d, horizon=0.3, color=(0,255,0)))
    plt.imshow(draw_patch(image,  15, -1.5, d, horizon=0.3, color=(255,0,255)))


def get_frame_maker(img):
    x = -15
    y = - 1.5
    def make_frame(t):
        frame = np.copy(img)
        z = 2**(t % 5)*5
        draw_patch(frame, -15, y, z, horizon=0.30, color=(0,255,255))
        draw_patch(frame, -7, y, z, horizon=0.30, color=(0,255,255))
        draw_patch(frame, 0, y, z, horizon=0.30, color=(0,255,255))
        draw_patch(frame, 7, y, z, horizon=0.30, color=(0,255,255))
        draw_patch(frame, 15, y, z, horizon=0.30, color=(0,255,255))
        return frame
    return make_frame


clip = VideoClip(get_frame_maker(mpimg.imread('bbox-example-image.jpg')), duration=5)
clip.write_videofile("output_images/patches1.mp4", fps=25)

a = cycle(np.mgrid[-10:10:5,2:100:30].T.reshape(-1,2))
