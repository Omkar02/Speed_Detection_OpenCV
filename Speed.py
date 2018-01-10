import cv2
import time
import numpy as np
#**********************************************************************************
firstFrame = cv2.imread('frame5.jpg')
cap = cv2.VideoCapture('Alibi ALI-IPU3030RV IP Camera Highway Surveillance.mp4')
#**********************************************************************************
kernel = np.ones((7,7),np.uint8)
kernal_erode = np.ones((4,4),np.uint8)
distance = 50                                   # assumed in meters
#**********************************************************************************
mask1 = cv2.imread('m1.jpg')
mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
ret1,thresh_MASK_1 = cv2.threshold(mask1,127,255,cv2.THRESH_BINARY_INV)
#********************************************
mask2 = cv2.imread('m2.jpg')
mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
ret2,thresh_MASK_2 = cv2.threshold(mask2,127,255,cv2.THRESH_BINARY_INV)
#********************************************
mask3 = cv2.imread('m3.jpg')
mask3 = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
ret3,thresh_MASK_3 = cv2.threshold(mask3,127,255,cv2.THRESH_BINARY_INV)
#**********************************************************************************
ti = 0
ti1 = 0
ti3 = 0
font = cv2.FONT_HERSHEY_SIMPLEX
#**********************************************************************************
while(cap.isOpened()):
    ret, frame = cap.read()
    gray1 = cv2.GaussianBlur(frame, (3, 3), 0)
    frameDelta = cv2.absdiff(firstFrame, frame)
    delta_gray = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)
    erode = cv2.erode(delta_gray, kernal_erode, iterations=2)
    dilation = cv2.dilate(erode, kernel, iterations=5)
    thresh = cv2.threshold(dilation, 25, 255, cv2.THRESH_BINARY)[1]
    thresh1 = np.bitwise_and(thresh, thresh_MASK_1)
    thresh2 = np.bitwise_and(thresh, thresh_MASK_2)
    thresh3 = np.bitwise_and(thresh, thresh_MASK_3)
# **********************************************************************************

    im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []
    height, width, _ = frame.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x, y, w, h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
        if w > 60 and h > 60:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.line(frame, (190, 311), (700, 311), (0, 255, 0), 3)
        cv2.line(frame, (90, 450), (800, 450), (0, 255, 0), 3)
        start_time = time.clock()
        if y > 311 :
            ti += 1
            if  y > 450:
                seconds = time.clock() - start_time
                #print('sec=', (seconds))
                seconds1 = (1.000275017283666 + float(seconds))
                speed = ((distance) / (ti * 2.4 * (seconds1))) * (3600 / 1000)
                a = print('Speed =', int(speed), 'Km/Hr')
                ti =0
            cv2.putText(frame, a, (50, 50), font, 6, (200, 255, 155), 13, )
# **********************************************************************************
    im1, contours1, hierarchy1 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        hierarchy1 = hierarchy1[0]
    except:
        hierarchy1 = []
    height1, width1, _ = frame.shape
    min_x1, min_y1 = width1, height1
    max_x1 = max_y1 = 0
    # computes the bounding box for the contour, and draws it on the frame,
    for contour1, hier1 in zip(contours1, hierarchy1):
        (x1, y1, w1, h1) = cv2.boundingRect(contour1)
        min_x1, max_x1 = min(x1, min_x1), max(x1 + w1, max_x1)
        min_y1, max_y1 = min(y1, min_y1), max(y1 + h1, max_y1)
        if w1 > 60 and h1 > 60:
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

        cv2.line(frame, (190, 311), (700, 311), (0, 0, 255), 3)
        cv2.line(frame, (90, 450), (800, 450), (0, 0, 255), 3)
        start_time1 = time.clock()
        if y1 > 311:
            ti1 += 1
            if y1 > 450:
                seconds1 = time.clock() - start_time1
                # print('sec=', (seconds))
                seconds1 = (1.000275017283666 + float(seconds1))
                speed1 = ((distance) / (ti1 * 2.4 * (seconds1))) * (3600 / 1000)
                a = print('Speed1 =', int(speed1), 'Km/Hr')
                ti1 = 0
                start_time1 = 0
# **********************************************************************************
        im3, contours3, hierarchy3 = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            hierarchy3 = hierarchy3[0]
        except:
            hierarchy3 = []
        height3, width3, _ = frame.shape
        min_x3, min_y3 = width1, height1
        max_x3 = max_y3 = 0
        # computes the bounding box for the contour, and draws it on the frame,
        for contour3, hier3 in zip(contours3, hierarchy3):
            (x3, y3, w3, h3) = cv2.boundingRect(contour3)
            min_x3, max_x3 = min(x3, min_x3), max(x3 + w3, max_x3)
            min_y3, max_y3 = min(y3, min_y3), max(y3 + h3, max_y3)
            if w3 > 60 and h3 > 60:
                cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (255, 0, 0), 2)

            cv2.line(frame, (190, 311), (700, 311), (0, 0, 255), 3)
            cv2.line(frame, (90, 450), (800, 450), (0, 0, 255), 3)
            start_time3 = time.clock()
            if y3 > 311:
                ti3 += 1
                if y3 > 450:
                    seconds3 = time.clock() - start_time3
                    # print('sec=', (seconds))
                    seconds3 = (1.000275017283666 + float(seconds3))
                    speed3 = ((distance) / (ti3 * 2.4 * (seconds3))) * (3600 / 1000)
                    a = print('Speed2 =', int(speed3), 'Km/Hr')
                    ti3 = 0
                    start_time3 = 0
# **********************************************************************************
    cv2.imshow('Original', dilation )
    cv2.imshow('Processed',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

