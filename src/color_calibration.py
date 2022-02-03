from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty
import cv2
import glob
import numpy as np
import imutils

def get_region_mean(img,point,radius):
    x, y = point
    samples = img[y-radius:y+radius, x-radius:x+radius]
    (rsquare, gsquare,bsquare) = cv2.split(samples)
    r, g, b = [int(rsquare.mean()),int(gsquare.mean()),int(bsquare.mean())]
    return (r,g,b)

def white_balance(img,x,y,radius):

    total = np.array([0,0,0])
    arr = [2,3]
    for i in arr:
        total = total + get_region_mean(img,(x[i],y[i]),radius)
    
    #grey = [((total/len(x)).mean())]*3
    grey = [((total/len(arr)).mean())]*3
    delta = (grey - total/len(arr)).astype(int)
    
    return np.clip(img.astype(int) + delta,0,255).astype('uint8')

def intensity_adjust(img,x,y,radius):
    graytones = [243,200,161,120,85,52]
    gray = []
    print(x,graytones,y)

    for i in range(len(x)):
        gray.append(int(np.array(get_region_mean(img,(x[i],y[i]),radius)).mean()))
    
    z = np.polyfit(gray,graytones,3)
    p = np.poly1d(z)

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pgray = np.clip(p(grayimg),0,255)
    delta = np.clip(pgray.astype(int) - grayimg.astype(int) + np.ones(grayimg.shape)*255,0,255*2)
    deltacolor = cv2.cvtColor(delta.astype('uint16'), cv2.COLOR_GRAY2RGB) 
    finalimg = np.clip(img.astype(int) + deltacolor.astype(int) - (np.ones(img.shape)*255).astype(int),0,255)

    return finalimg.astype('uint8')
    
def get_coords(image_path, template_path, template_coords,show = False):
    template = cv2.imread(template_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    if show:
        cv2_imshow(template)

    for imagePath in glob.glob(image_path):

        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None

        for scale in np.linspace(0.025, 0.5, 20)[::-1]:

            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])

            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            if True:

                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                  (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                if show:    
                    cv2_imshow(clone)
                    cv2.waitKey(0)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

        if show:
            cv2_imshow(image)
        cv2.waitKey(0)

    path = image_path
    image = cv2.imread(path)
    radius = int(image.shape[0]*0.075)
    color = (255, 0, 0)
    thickness = int(image.shape[0]*0.01)
    pallete_coords = []

    for coord in template_coords:
        x = int(coord[0]/152 * (endX - startX))
        y = int(coord[1]/101 * (endY - startY))
        x = x + startX
        y = y + startY
        coord = (x,y)
        if show:
            image = cv2.circle(image, coord, radius, color, thickness)
        pallete_coords.append((x,y))

    if show:
        cv2_imshow(image)

    return pallete_coords
 	
def calibrate(img,pallete_coords,radius=5):
	x = []
	y = []
	for xy in pallete_coords:
		x.append(xy[0])
		y.append(xy[1])
	img = white_balance(img,x,y,radius)
	img = intensity_adjust(img,x,y,radius)
	return img
 	
def draw_circles(img, coords):
	#radius = int(img.shape[0]*0.025)
	#color = (255, 0, 0)
	thickness = int(img.shape[0]*0.025)
	for coord in coords:
		img = cv2.circle(img, coord, radius=0, color=(0,255,255), thickness=thickness)        
	return img
	
