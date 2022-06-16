import cv2
import matplotlib.pyplot as plt
import numpy as np

def hullarea(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

def remove_one_vertex(hull):
    new_hull = []
    values = {}
    for i in range(len(hull)):
        test = []
        for j in range(len(hull)):
            if j != i:
                test.append(hull[j])
        values[hullarea(test)] = test
    return np.array(values[max(list(values.keys()))])

def quality_control(hull,thresh=0.75):
    dists = []
    for i in hull:
        for j in hull:
             dists.append(np.linalg.norm(i-j))
    a = np.mean(dists)
    c = (a*a)/hullarea(hull)

    if (c > thresh):
        return None

    if a < 16:
        return None
    return hull

def quadrilateralize(hull):
    if len(hull) < 4:
        return None
    else:
        while(len(hull) != 4):
            hull = remove_one_vertex(hull)
    return quality_control(hull)

def get_rects(img,use_canny=True):

    if use_canny:
        img = cv2.Canny(img,32,32)
        cv2.imwrite("canny.jpg",img)
        img = cv2.imread("canny.jpg")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    paint_img = img.copy()
    rects = []

    for i in range(len(cnts)):
        hull = cv2.convexHull(cnts[i])
        new_hull = []
        for i in hull:
            new_hull.append(i[0])

        hull = quadrilateralize(new_hull)

        if hull is not None:

            #cv2.drawContours(paint_img, np.array([hull]), -1, (0, 0, 255), 2)
            rects.append(hull)

    #cv2.imwrite("salvo.png",paint_img)
    #plt.imshow(paint_img)
    return rects

def filter(rects,img):
    paint_img = img.copy()

    areas = []
    for rect in rects:
        areas.append(hullarea(rect))
    mode = sorted(areas)[int(len(areas)/2)]
    new_rects = []
    for rect in rects:
        t = 0.5
        area = hullarea(rect)
        if ((area/mode < 1/t) and (area/mode > t)):
            new_rects.append(rect)
            #cv2.drawContours(paint_img, np.array([rect]), -1, (0, 255, 0), 2)
    #plt.imshow(paint_img)
    return new_rects

def get_pallete_box(rects,img):
    paint_img = img.copy()

    points = []

    for rect in rects:
        for point in rect:
            points.append(point)

    cnt = cv2.convexHull(np.array(points))

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(paint_img,[box],0,(0,255,0),5)

    #plt.imshow(paint_img)
    return box

def organize(a):
    a = a[a[:, 0].argsort()]
    a = a[:, a[0, :].argsort()]
    return a

def homography(rect,img):

    #rect = organize(rect)
    size = 128
    width = size
    height = size

    srcpts = np.float32(rect)
    destpts = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
    resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
    resultimage = cv2.warpPerspective(img, resmatrix, (width, height))

    plt.imshow(resultimage)
    #cv2.imwrite("debug_pallete.jpg",resultimage)
    return resultimage

def get_palette(img):
    rects = get_rects(img)
    rects = filter(rects,img)
    rect = get_pallete_box(rects,img)
    palette = homography(rect,img)
    return palette, rect

def resize(img):
    maxsize = 640
    img = cv2.resize(img, (maxsize,int(img.shape[0]*maxsize/img.shape[1])))
    return img


def get_color_points(img):
    img = resize(img)
    pallete, rect = get_palette(img)

    colors = [[10,10,10],[20,20,20],[10,10,10]]
    gray = []
    for i in colors:
        gray.append(int(np.mean(i)))

    cont = 0
    while ((gray != sorted(gray)) and cont < 4):
        pallete = cv2.rotate(pallete, cv2.ROTATE_90_CLOCKWISE)
        points = []
        for i in range(1,7):
            x = int(pallete.shape[1]/6*(i-0.5))
            y = int(pallete.shape[0]/6*5.25)
            points.append([x,y])

        colors = []
        for i in points:
            colors.append(pallete[i[1]][i[0]][:])

        gray = []
        for i in colors:
            gray.append(int(np.mean(i)))
        cont = cont + 1

    #cv2.imwrite("pallete.jpg",pallete)
    return np.array(colors), rect

def white_balance(img,pallete_colors):

    total = np.array([0,0,0])
    arr = [2,3]

    for i in arr:
        total = total + pallete_colors[i]

    #grey = [((total/len(x)).mean())]*3
    grey = [((total/len(arr)).mean())]*3
    delta = (grey - total/len(arr)).astype(int)

    return np.clip(img.astype(int) + delta,0,255).astype('uint8')

def intensity_adjust(img,pallete_colors):
    gray = []

    graytones = [52,85,120,161,200,243]

    for i in range(len(pallete_colors)):
        gray.append(int(np.array(pallete_colors[i]).mean()))

    z = np.polyfit(gray,graytones,3)
    p = np.poly1d(z)

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pgray = np.clip(p(grayimg),0,255)
    delta = np.clip(pgray.astype(int) - grayimg.astype(int) + np.ones(grayimg.shape)*255,0,255*2)
    deltacolor = cv2.cvtColor(delta.astype('uint16'), cv2.COLOR_GRAY2RGB)
    finalimg = np.clip(img.astype(int) + deltacolor.astype(int) - (np.ones(img.shape)*255).astype(int),0,255)

    return finalimg.astype('uint8')


def debug(img,rect):
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2RGB)
    debug_img = resize(img.copy())
    debug_img = cv2.drawContours(debug_img, np.array([rect]), -1, (167,207,120), 12)
    return debug_img

def calibrate_batch(ref,imgs):

    results = []
    samples, rect = get_color_points(ref)
    print("PRINT DOS SAMPLES",samples)

    debug_img = debug(ref,rect)
    #cv2.imwrite("debug_find_pallete.jpg",debug_img)

    for i in imgs:
        img = i.copy()
        img = white_balance(img,samples)
        img = intensity_adjust(img,samples)
        results.append(img)

    return results, debug_img

#def calibrate(img):
#    cv2.imwrite("imagem_de_debug_dia_14_antes.jpg",img)
#    samples, rect = get_color_points(img)
#    img = white_balance(img,samples)
#    img = intensity_adjust(img,samples)
#    cv2.imwrite("imagem_de_debug_dia_14_depois.jpg",img)
#    return img, rect
