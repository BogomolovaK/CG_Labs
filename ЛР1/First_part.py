from math import cos, sqrt
from math import sin
import numpy as np
from PIL import Image

def dotted_line0(image, x0, y0, x1, y1, count, color):
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round ((1.0 - t) * x0 + t * x1)
        y = round ((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def dotted_line(image, x0, y0, x1, y1, color):
    count = sqrt((x0 - x1)**2+(y0 - y1)**2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round ((1.0 - t) * x0 + t * x1)
        y = round ((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def x_loop_line0(image, x0, y0, x1, y1, color):
    for x in range (x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round ((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def x_loop_line(image, x0, y0, x1, y1, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    for x in range (x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round ((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def draw_line0(image, x0, y0, x1, y1, color):
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0: return
    x, y = x0, y0
    for i in range(steps + 1):
        image[int(round(y)), int(round(x))] = color
        x += (x1 - x0) / steps
        y += (y1 - y0) / steps

def draw_line1(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range (x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update

def Brenhem(image, x0, y0, x1, y1, color):
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

img_mat = np.zeros((200, 200, 3), dtype = np.uint8)
color = [0, 203, 254]


for i in range(13):
    x0 = 100
    y0 = 100
    x1 = int(100 + 95*cos((i*2*3.14)/13))
    y1 = int(100 + 95*sin((i*2*3.14)/13))
    #dotted_line0(img_mat, x0,y0,x1,y1, color)
    #dotted_line(img_mat, x0,y0,x1,y1, color)
    #x_loop_line0(img_mat,x0,y0,x1,y1,color)
    #x_loop_line(img_mat,x0,y0,x1,y1,color)
    #draw_line0 (img_mat, x0, y0, x1, y1, color)
    #draw_line1 (img_mat, x0, y0, x1, y1, color)
    Brenhem(img_mat,x0,y0,x1,y1,color)
img = Image.fromarray(img_mat, mode = 'RGB')
img.save('img.png')