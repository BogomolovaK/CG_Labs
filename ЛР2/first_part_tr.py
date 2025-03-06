from math import cos, sqrt
from math import sin
import numpy as np
from random import randint
from PIL import Image, ImageOps

def bary(x0, y0, x1, y1, x2, y2, x, y):
    l0 = ((x - x2) * (y1 - y2) - (x1 - x2)*(y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 -y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l2 = 1.0 - l0 - l1
    return l0, l1, l2

def draw_tr_base (image, x0, y0, x1, y1, x2, y2, color):
    xmin = int(np.floor(max(min(x0, x1, x2), 0)))
    ymin = int(np.floor(max(min(y0, y1, y2), 0)))

    xmax = int(np.ceil(min(max(x0, x1, x2), img_size)))
    ymax = int(np.ceil(min(max(y0, y1, y2), img_size)))

    for x in range (xmin, xmax):
        for y in range (ymin, ymax):
            l0, l1, l2 = bary(x0, y0, x1, y1, x2, y2, x, y)
            if (l0 >= 0 and l1 >= 0 and l2 >= 0):
                image[x, y] = color

def draw_tr_rand (image, x0, y0, x1, y1, x2, y2):
    xmin = int(np.floor(max(min(x0, x1, x2), 0)))
    ymin = int(np.floor(max(min(y0, y1, y2), 0)))

    xmax = int(np.ceil(min(max(x0, x1, x2), img_size)))
    ymax = int(np.ceil(min(max(y0, y1, y2), img_size)))

    color = [randint(0, 255), randint(0, 255),randint(0, 255)]

    for x in range (xmin, xmax):
        for y in range (ymin, ymax):
            l0, l1, l2 = bary(x0, y0, x1, y1, x2, y2, x, y)
            if (l0 >= 0 and l1 >= 0 and l2 >= 0):
                image[x, y] = color

def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    normal = np.cross ([x1 - x2, y1 - y2, z1 - z2], [x1-x0, y1-y0, z1-z0])
    return normal

def find_light(x0, y0,z0, x1, y1, z1, x2, y2, z2):
    l = [0, 0, 1]
    n = normal(x0, y0,z0, x1, y1, z1, x2, y2, z2)
    norm_n = np.linalg.norm(n)
    norm_l = np.linalg.norm(l)
    cos_light = (np.dot(n, l)) / (norm_n * norm_l)
    return cos_light

def draw_tr_with_light (image, x0, y0, z0, x1, y1, z1, x2, y2, z2):
    xmin = int(np.floor(max(min(x0, x1, x2), 0)))
    ymin = int(np.floor(max(min(y0, y1, y2), 0)))

    xmax = int(np.ceil(min(max(x0, x1, x2), img_size)))
    ymax = int(np.ceil(min(max(y0, y1, y2), img_size)))

    for x in range (xmin, xmax):
        for y in range (ymin, ymax):
            l0, l1, l2 = bary(x0, y0, x1, y1, x2, y2, x, y)
            light = find_light(x0, y0, z0, x1, y1, z1, x2, y2, z2)
            if (l0 >= 0 and l1 >= 0 and l2 >= 0 and light>=0):
                color = [0, 203*light, 254*light]
                image[x, y] = color
                
def draw_tr (image, x0, y0, z0, x1, y1, z1, x2, y2, z2):
    xmin = int(np.floor(max(min(x0, x1, x2), 0)))
    ymin = int(np.floor(max(min(y0, y1, y2), 0)))

    xmax = int(np.ceil(min(max(x0, x1, x2), img_size)))
    ymax = int(np.ceil(min(max(y0, y1, y2), img_size)))

    light = find_light(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    if (light == 0):
        return
    color = [0, -203*light, -254*light]

    for x in range (xmin, xmax):
        for y in range (ymin, ymax):
            l0, l1, l2 = bary(x0, y0, x1, y1, x2, y2, x, y)
        
            if (l0 >= 0 and l1 >= 0 and l2 >= 0):
                z = l0*z0 + l1*z1 + l2*z2
                if z < z_buffer[x, y]:
                    image[x, y] = color
                    z_buffer[x, y] = z
                

img_size = 2000
img_mat = np.zeros((img_size, img_size, 3), dtype = np.uint8)
z_buffer = np.zeros((img_size, img_size), dtype = np.float32)
z_buffer[:] = np.inf

v = []
f = []

obj_file = open('model_1.obj')
for line in obj_file:
    s = line.split()
    if (s[0] == 'v'): v.append([float(s[1]), float(s[2]), float(s[3])])
    if (s[0] == 'f'): f.append([int(s[1].split('/')[0]),int( s[2].split('/')[0]), int(s[3].split('/')[0])])

for i in range(len (f)):
    x0 = (v[f[i][0]-1][0]*5000*2+500*2)
    y0 = (v[f[i][0]-1][1]*5000*2+500*2)
    z0 = (v[f[i][0]-1][2]*5000*2+500*2)

    x1 = (v[f[i][1]-1][0]*5000*2+500*2)
    y1 = (v[f[i][1]-1][1]*5000*2+500*2)
    z1 = (v[f[i][1]-1][2]*5000*2+500*2)

    x2 = (v[f[i][2]-1][0]*5000*2+500*2)
    y2 = (v[f[i][2]-1][1]*5000*2+500*2)
    z2 = (v[f[i][2]-1][2]*5000*2+500*2)

    draw_tr (img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2)

img = Image.fromarray(img_mat, mode = 'RGB')
img.save('img13.png')