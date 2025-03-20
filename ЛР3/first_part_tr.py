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

def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    normal = np.cross ([x1 - x2, y1 - y2, z1 - z2], [x1-x0, y1-y0, z1-z0])
    return normal

def find_light(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    l = [0, 0, 1]
    n = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    norm_n = np.linalg.norm(n)
    norm_l = np.linalg.norm(l)
    cos_light = (np.dot(n, l)) / (norm_n * norm_l)
    return cos_light

                
def draw_tr (image, x0, y0, z0, x1, y1, z1, x2, y2, z2):

    x0_p = a*x0/z0 + img_size/2
    y0_p = a*y0/z0 + img_size/2

    x1_p = a*x1/z1 + img_size/2
    y1_p = a*y1/z1 + img_size/2

    x2_p = a*x2/z2 + img_size/2
    y2_p = a*y2/z2 + img_size/2
    
    xmin = int(np.floor(max(min(x0_p, x1_p, x2_p), 0)))
    ymin = int(np.floor(max(min(y0_p, y1_p, y2_p), 0)))

    xmax = int(np.ceil(min(max(x0_p, x1_p, x2_p), img_size)))
    ymax = int(np.ceil(min(max(y0_p, y1_p, y2_p), img_size)))

    light = find_light(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    if (light == 0):
        return
    color = [0, -203*light, -254*light]

    for x in range (xmin, xmax):
        for y in range (ymin, ymax):
            l0, l1, l2 = bary(x0_p, y0_p, x1_p, y1_p, x2_p, y2_p, x, y)
        
            if (l0 >= 0 and l1 >= 0 and l2 >= 0):
                z = l0*z0 + l1*z1 + l2*z2
                if z < z_buffer[x, y]:
                    image[x, y] = color
                    z_buffer[x, y] = z

def row(x,y,z):
    a = 90
    b = 90
    u = 0
    n = [x, y, z]
    rx = [[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]]
    ry = [[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]]
    rz = [[cos(u), sin(u), 0], [-sin(u), cos(u), 0], [0, 0, 1]]
    temp = np.matmul(rx, ry)
    temp = np.matmul(temp, rz)
    temp = np.matmul(temp, n)
    x1 = temp[0] 
    y1 = temp[1] 
    z1 = temp[2] + tz
    return x1, y1, z1


tz = 0.1           
a = 10000*tz
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

for i in range(len (v)):
    v[i][0], v[i][1], v[i][2] = row (v[i][0], v[i][1], v[i][2])

for i in range(len (f)):

    x0 = (v[f[i][0]-1][0])
    y0 = (v[f[i][0]-1][1])
    z0 = (v[f[i][0]-1][2])

    x1 = (v[f[i][1]-1][0])
    y1 = (v[f[i][1]-1][1])
    z1 = (v[f[i][1]-1][2])

    x2 = (v[f[i][2]-1][0])
    y2 = (v[f[i][2]-1][1])
    z2 = (v[f[i][2]-1][2])

    draw_tr (img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2)

img = Image.fromarray(img_mat, mode = 'RGB')
img.save('try9.png')