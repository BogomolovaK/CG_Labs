from math import cos, sqrt
from math import sin
import numpy as np
from PIL import Image, ImageOps

def draw_line(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        x, y = round(x), round(y)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)
color = [0, 203, 254]
v = []
f = []

obj_file = open('model_1.obj')
for line in obj_file:
    s = line.split()
    if (s[0] == 'v'): v.append([float(s[1]), float(s[2]), float(s[3])])
    if (s[0] == 'f'): f.append([int(s[1].split('/')[0]),int( s[2].split('/')[0]), int(s[3].split('/')[0])])


for i in range(len (f)):
    x0 = int(v[f[i][0]-1][0]*5000*2+500*2)
    y0 = int(v[f[i][0]-1][1]*5000*2+500*2)

    x1 = int(v[f[i][1]-1][0]*5000*2+500*2)
    y1 = int(v[f[i][1]-1][1]*5000*2+500*2)

    x2 = int(v[f[i][2]-1][0]*5000*2+500*2)
    y2 = int(v[f[i][2]-1][1]*5000*2+500*2)

    draw_line(img_mat, x0, y0, x1, y1, color)
    draw_line(img_mat, x1, y1, x2, y2, color)
    draw_line(img_mat, x2, y2, x0, y0, color)
    
img = Image.fromarray(img_mat, mode = 'RGB')
img.save('img3.png')