import numpy
import numpy as np
import matplotlib
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
from matplotlib.pyplot import cm
import io
from get_color import get_bar_color,get_line_color
# load the image
color = get_bar_color()
line = get_line_color()


def covert_to_grey(image_name):
    image = Image.open('/nas/yichung/weather/data/' + image_name)
    data = asarray(image)
    img = (numpy.copy(data)).tolist()
    
    for a in range(2700):
        if(a%100==0):
            print(a//100)
        for b in range(3600):
            if(img[a][b] not in color):
                img[a][b] = 255
            else:
                img[a][b] = int(color.index(img[a][b]))

    img = (np.asarray(img))*3
    
        
    img[img>255]=255
    raw = (numpy.copy(data)).tolist()

    for a in range(2700):
        for b in range(530,3150):
            if(not ((raw[a][b] in color) or (raw[a][b] in [[255,255,255],[229,229,229],[243,245,248]]))):
                pixel_list = []
                for c in range(-3,4):
                    for h in range(-3,4):
                        try:
                            if raw[a+c][b+h] in color or (raw[a+c][b+h] in [[255,255,255],[229,229,229]]):
                                pixel_list.append(img[a+c][b+h])
                        except:
                            pass
                if len(pixel_list)!=0:
                    img[a][b] = int(sum(pixel_list)/len(pixel_list))
    im = asarray(img)
    imfile = io.BytesIO()
    imsave('./pic/grey_'+image_name, im, format="png", cmap=cm.gray)
    
    
def covert_to_grey_3600(image_name):
    image = Image.open('/nas/yichung/weather/data/' + image_name)
    data = asarray(image)
    img = (numpy.copy(data)).tolist()
    
    for a in range(3600):
        for b in range(3600):
            if(img[a][b] not in color):
                img[a][b] = 255
            else:
                img[a][b] = int(color.index(img[a][b]))

    img = (np.asarray(img))*3
    
        
    img[img>255]=255
    raw = (numpy.copy(data)).tolist()

    for a in range(3600):
        if(a%100==0):
            print(a//100)
        for b in range(530,3150):
            if(not ((raw[a][b] in color) or (raw[a][b] in [[255,255,255],[229,229,229],[243,245,248]]))):
                pixel_list = []
                for c in range(-3,4):
                    for h in range(-3,4):
                        try:
                            if raw[a+c][b+h] in color or (raw[a+c][b+h] in [[255,255,255],[229,229,229]]):
                                pixel_list.append(img[a+c][b+h])
                        except:
                            pass
                if len(pixel_list)!=0:
                    img[a][b] = int(sum(pixel_list)/len(pixel_list))
    im = asarray(img)
    imfile = io.BytesIO()
    imsave('./pic3600/grey_'+image_name, im, format="png", cmap=cm.gray)
