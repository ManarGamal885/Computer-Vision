from time import sleep
from tracker import *
import os
import datetime as dt
import cv2 as cv
import numpy as np
import cv2
import os
os.chdir('E:\computerVision\projects')
img = cv2.imread('E:\computerVision\pic.jpg', 0)
print(img)

cv2.imshow('firstImage', img)
k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('secondImage.jpg', img)
    cv2.destroyAllWindows()
# --------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
img = cv2.imread('E:\computerVision\pic.jpg', 1)
print(img)

cv2.imshow('firstImage', img)
k = cv2.waitKey(5000)
# ---------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
print(cap.isOpened())
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out.write(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
# ----------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
img = np.zeros([512, 512, 3], np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
img = cv2.polylines(img, [pts], True, (0, 255, 255))
# img = cv2.putText(img,'Manar',(10,210),font,4,(0,255,0),10)
# img = cv2.line(img, (0, 0), (256, 256), (0, 255, 0), 10)
# img = cv2.arrowedLine(img ,(0,255),(256,256),(255,0,0),5)
cv2.imshow('image', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
# ------------------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
img = np.zeros([600, 900, 3], np.uint8)
cv.rectangle(img, (0, 0), (900, 500), (255, 225, 85), -1)
cv.rectangle(img, (0, 500), (900, 600), (75, 180, 70), -1)
cv.circle(img, (200, 150), 60, (0, 255, 255), -1)
cv.circle(img, (200, 150), 75, (220, 255, 255), 1)
cv.line(img, (710, 500), (710, 420), (30, 65, 155), 15)
triangleTree = np.array([[640, 460], [780, 460], [710, 300]], np.int32)
cv.fillPoly(img, [triangleTree], (75, 180, 70))
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
# ------------------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(3, 3000)
cap.set(4, 3000)
print(cap.get(3))
print(cap.get(4))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
# -------------------------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(3, 3000)
cap.set(4, 3000)
print(cap.get(3))
print(cap.get(4))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ' Width: ' + str(cap.get(3))+' Height: '+str(cap.get(4))
        dateTime = str(dt.datetime.now())

        frame = cv2.putText(frame, text, (10, 50), font, 1, (0, 2555, 255), 4)
        frame = cv2.putText(frame, dateTime, (20, 100),
                            font, 1, (0, 2555, 255), 4)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
# ----------------------------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
img = cv2.imread('E:\computerVision\pic.jpg')


def clickListner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_TRIPLEX
        strXY = str(x) + ' ' + str(y)
        cv2.putText(img, strXY, (x, y), font, 1, (255, 255, 0), 4)
        cv2.imshow('Image', img)
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_TRIPLEX
        strRGB = str(red) + ' ' + str(green) + ' ' + str(blue)
        cv2.putText(img, strRGB, (x, y), font, 1, (0, 255, 255), 4)
        cv2.imshow('Image', img)


cv2.imshow('Image', img)
cv2.setMouseCallback('Image', clickListner)
cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
def clickListner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 15, (0, 0, 255), -1)
        points.append((x, y))
        if len(points) >= 2:
            cv2.line(img, points[-1], points[-2], (0, 255, 255))
        cv2.imshow('Image', img)


img = np.zeros((512, 512, 3), np.int8)
cv2.imshow('Image', img)
points = []
cv2.setMouseCallback('Image', clickListner)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
# focusing on a specifig part of a photo
# os.chdir('adding a specific path')
img = cv.imread("E:\computerVision\pic.jpg")
img2 = cv.imread("E:\computerVision\messi.jpg")
print(img.shape)
print(img.size)
print(img.dtype)
b, g, r = cv.split(img)
img = cv.merge((b, g, r))
ball = img2[390:460, 430:510]
img2[393:463, 90:170] = ball
# add to photos to one photo
img = cv.resize(img, (512, 512))
img2 = cv.resize(img2, (512, 512))
dst = cv.add(img, img2)
dst = cv.addWeighted(img, .8, img2, .2, 1)
cv.imshow('Img', dst)
cv.waitKey(0)
cv.destroyAllWindows()
# -------------------------------------------
# bitwise oper
# don
# -------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow('image')


def nothing(x):
    print(x)


cv.createTrackbar('B', 'image', 0, 255, nothing)
cv.createTrackbar('R', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
switch = '0:OFF\n 1:ON'
cv.createTrackbar(switch, 'image', 0, 1, nothing)
while (1):
    cv.imshow('image', img)
    k = cv.waitKey(1)
    if k == 27:
        break
    b = cv.getTrackbarPos('B', 'image')
    r = cv.getTrackbarPos('R', 'image')
    g = cv.getTrackbarPos('G', 'image')
    s = cv.getTrackbarPos(switch, 'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, r, g]
cv.destroyAllWindows()
# ----------------------------------------------
while (True):
    frame = cv.imread("E:\computerVision\smarties.png")
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    l_b = np.array([110, 50, 50])
    u_b = np.array([130, 255, 255])

    mask = cv.inRange(hsv, l_b, u_b)
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow("detecting the blue ball", frame)
    cv.imshow("mask", mask)
    cv.imshow("res", res)

    k = cv.waitKey(0)
    if k == 27:
        break
cv.destroyAllWindows()
# ------------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')


def nothing(x):
    pass


cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)


while True:

    frame = cv2.imread("E:\computerVision\smarties.png")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
# ------for vid-------------------'
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')


def nothing(x):
    print(x)


cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

cap = cv2.VideoCapture(0)
while True:

    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
# ------------color detection--------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape

    cx = int(width / 2)
    cy = int(height / 2)

    # Pick pixel value
    pixel_center = hsv_frame[cy, cx]
    hue_value = pixel_center[0]

    color = "Undefined"
    if hue_value < 5:
        color = "RED"
    elif hue_value < 22:
        color = "ORANGE"
    elif hue_value < 33:
        color = "YELLOW"
    elif hue_value < 78:
        color = "GREEN"
    elif hue_value < 131:
        color = "BLUE"
    elif hue_value < 170:
        color = "VIOLET"
    else:
        color = "RED"

    pixel_center_bgr = frame[cy, cx]
    b, g, r = int(pixel_center_bgr[0]), int(
        pixel_center_bgr[1]), int(pixel_center_bgr[2])

    cv2.rectangle(frame, (cx - 220, 10), (cx + 200, 120), (255, 255, 255), -1)
    cv2.putText(frame, color, (cx - 200, 100), 0, 3, (b, g, r), 5)
    cv2.circle(frame, (cx, cy), 5, (25, 25, 25), 3)
    cv2.imshow("Frame", frame)
    # out.write(frame) #save your video
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()
# -----------------------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
cap = cv2.VideoCapture(0)
frameW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # '*XVID'
out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # applaying the filter we will use Gusssian filter
    # fillter (5,5)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # applaying theeshold to then applay the counturing
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
    # we make then dilation":expand for pixels
    # large iteration mwans large dilation
    dilated = cv2.dilate(thresh, None, iterations=10)
    # contures to spesifay the edges
    countours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for countour in countours:
        (x, y, w, h) = cv2.boundingRect(countour)
        if cv2.contourArea(contour=countour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "status of object: {}".format('moving'),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    img = cv2.resize(frame1, (1280, 720))
    out.write(img)
    cv2.imshow('motion dectection', frame1)

    frame1 = frame2
    _, frame2 = cap.read()

    if cv2.waitKey(100) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()
# ------------------------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')
tracker = EuclideanDistTracker()
cap = cv2.VideoCapture('highway.mp4')

# algo for detection
# varThreshold to remove noise
object_detector = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=50)

while True:
    ret, frame = cap.read()
    H, W, _ = frame.shape
    regonOfIn = frame[340:720, 500:800]
    mask = object_detector.apply(regonOfIn)
    # apply threasho to remove shdow
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # countouring
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # dectections to keep track of the objects while moving
    detections = []
    for contor in contours:
        area = cv2.contourArea(contor)
        if area > 100:
            (x, y, w, h) = cv2.boundingRect(contor)

            detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)
    for box in boxes_ids:
        x, y, w, h, id = box
        cv2.putText(regonOfIn, str(id), (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.rectangle(regonOfIn, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("reagonOfInterset", regonOfIn)
    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)

    if cv2.waitKey(30) == 27:
        break
cap.release()
cv2.destroyAllWindows()


# ----------------------------------------------------------
import cv2
import numpy as np
from time import sleep
import os
os.chdir('E:\computerVision\projects')

largura_min = 80
altura_min = 80
offset = 6
pos_linha = 550
delay = 60
detec = []
carros = 0


def pega_centro(x, y, w, h):

    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# video source input
cap = cv2.VideoCapture('video.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()
#ymin ymax, xmin, xmax

while True:
    
    ret, frame1 = cap.read()
    regonOfIn = frame1[348:353, 469:469]
    tempo = float(1/delay)

    sleep(tempo)

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = subtracao.apply(blur)

    dilat = cv2.dilate(img_sub, np.ones((5, 5)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dilatada = cv2.morphologyEx(dilat, cv2. MORPH_CLOSE, kernel)

    dilatada = cv2.morphologyEx(dilatada, cv2. MORPH_CLOSE, kernel)

    contorno, h = cv2.findContours(
        dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (176, 130, 39), 2)

    for (i, c) in enumerate(contorno):

        (x, y, w, h) = cv2.boundingRect(c)

        validar_contorno = (w >= largura_min) and (h >= altura_min)

        if not validar_contorno:

            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        centro = pega_centro(x, y, w, h)

        detec.append(centro)

        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

        for (x, y) in detec:

            if (y < (pos_linha + offset)) and (y > (pos_linha-offset)):

                carros += 1

                cv2.line(frame1, (25, pos_linha),
                         (1200, pos_linha), (0, 127, 255), 3)

                detec.remove((x, y))

                print("No. of cars detected : " + str(carros))

    cv2.putText(frame1, "VEHICLE COUNT : "+str(carros), (320, 70),cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
    cv2.imshow("Video Original", frame1)
    cv2.imshow(" Detectar ", dilatada)

    if cv2.waitKey(1) == 27:

        break
cv2.destroyAllWindows()

cap.release()
#---------------------------------------------

