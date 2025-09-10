import cv2
import numpy as np
import os
import hand_track as htm

# ------------------------------------
brushThickness = 15
eraserThickness = 100
# --------------------------------ca----

folderpath = "header"
# Ensure deterministic ordering of toolbar images (e.g., 1.jpg,2.jpg,...)
mylist = sorted(os.listdir(folderpath))
print(mylist)
overlaylist = []
for impath in mylist:
    image = cv2.imread(f'{folderpath}/{impath}')
    overlaylist.append(image)
print(len(overlaylist))

header = overlaylist[0]
drawColor = (255, 0, 255)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmlist = detector.findPositions(img, draw=False)

    if len(lmlist) !=0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        finger = detector.fingersUp()

        if finger[1] and finger[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)

            # Dynamic toolbar selection based on header image width & count
            header_h, header_w = header.shape[:2]
            if y1 < header_h:
                segment_w = header_w // len(overlaylist)
                idx = min(x1 // segment_w, len(overlaylist)-1)
                header = overlaylist[idx]
                # Map index to color (last one treated as eraser/black)
                palette = [ (255,0,255), (255,0,0), (0,255,0), (0,0,0) ]
                if idx < len(palette):
                    drawColor = palette[idx]

            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)
        if finger[1] and finger[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            # Simple smoothing: skip very small movements to reduce jitter
            if abs(x1 - xp) + abs(y1 - yp) < 4:
                continue
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imCanvas)

    img[0:125, 0:1280] = header
    cv2.imshow('Image', img)
    cv2.imshow('Canvas', imCanvas)
    cv2.waitKey(2)
