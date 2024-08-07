import cv2
import numpy as np
import os
import HandTrackingModule as htm
from collections import deque

#################
brushThickness = 15
eraseThickness = 50
#################

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawcolor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# History for undo/redo
canvasHistory = deque(maxlen=10)
redoHistory = deque(maxlen=10)


# Function to apply smoothing using moving average
def smooth_line(x1, y1, x2, y2, smoothing_factor=0.2):
    return int(x1 * (1 - smoothing_factor) + x2 * smoothing_factor), int(
        y1 * (1 - smoothing_factor) + y2 * smoothing_factor)


while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        # Tip of index and middle fingers:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        # 4. Erase all drawings if four fingers are up
        if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            canvasHistory.clear()
            redoHistory.clear()

        # 5. If selection mode - 2 fingers are up
        elif fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # Checking for click
            if y1 < 125:
                if 150 < x1 < 260:
                    header = overlayList[0]
                    drawcolor = (0, 0, 255)
                elif 360 < x1 < 470:
                    header = overlayList[1]
                    drawcolor = (255, 255, 0)
                elif 470 < x1 < 680:
                    header = overlayList[2]
                    drawcolor = (0, 255, 0)
                elif 790 < x1 < 910:
                    header = overlayList[3]
                    drawcolor = (0, 0, 0)
                #elif 920 < x1 < 1030:
                    #brushThickness = max(5, brushThickness - 5)
                #elif 1040 < x1 < 1150:
                    #brushThickness = min(50, brushThickness + 5)
                elif 1100 < x1 < 1270:
                    # Save the current drawing to the Pictures folder
                    save_path = os.path.join(os.path.expanduser('~'), 'Pictures', 'drawing.png')
                    cv2.imwrite(save_path, imgCanvas)
                    print(f"Drawing saved to {save_path}")
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawcolor, cv2.FILLED)

        # 6. If drawing mode - Index finger is up
        elif fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawcolor, cv2.FILLED)
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
                # Save the initial state for undo
                canvasHistory.append(imgCanvas.copy())

            # Smooth the line
            x1, y1 = smooth_line(xp, yp, x1, y1)

            if drawcolor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, eraseThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, eraseThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)

    key = cv2.waitKey(1)

    # Undo functionality
    if key == ord('z'):
        if len(canvasHistory) > 0:
            redoHistory.append(imgCanvas.copy())
            imgCanvas = canvasHistory.pop()

    # Redo functionality
    if key == ord('y'):
        if len(redoHistory) > 0:
            canvasHistory.append(imgCanvas.copy())
            imgCanvas = redoHistory.pop()

    # Exit if 'q' is pressed or the window is closed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()