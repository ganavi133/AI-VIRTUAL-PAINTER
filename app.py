from flask import Flask, render_template, Response, send_from_directory
import cv2
import os
import numpy as np
import uuid
import HandTrackingModule as htm

app = Flask(__name__)

# Constants for brush and eraser thickness
brushThickness = 15
eraserThickness = 50
circleThickness = 2  # Reduced thickness of the circle border
circle_radius = 100  # Increased radius of the circle
laser_radius = 5  # Radius of the laser pointer dot

# Folder path containing header images
folderPath = "header"
myList = os.listdir(folderPath)
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]
header = overlayList[0]  # Initial header image
drawColor = (255, 0, 255)  # Initial drawing color

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Folder to store screenshots
screenshot_folder = os.path.join(os.path.expanduser('~'), 'Pictures', 'PainterScreenshots')
os.makedirs(screenshot_folder, exist_ok=True)

# Initialize variables for dragging circle
dragging = False
x1_start, y1_start = 0, 0

# Variables for laser pointer
laser_x, laser_y = 0, 0

# Function to generate video frames
def generate_frames():
    global xp, yp, imgCanvas, header, drawColor, brushThickness, dragging, x1_start, y1_start, laser_x, laser_y

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]
                fingers = detector.fingersUp()

                # Update laser pointer position if middle finger is lifted
                if fingers[2]:
                    laser_x, laser_y = lmList[12][1:]

                # Draw a large circle when three fingers are up and dragging
                if fingers[1] and fingers[2] and fingers[3] and not fingers[0] and not fingers[4]:
                    if not dragging:
                        x1_start, y1_start = x1, y1
                        dragging = True
                    else:
                        cv2.circle(img, (x1_start, y1_start), circle_radius, drawColor, circleThickness)
                        cv2.circle(imgCanvas, (x1_start, y1_start), circle_radius, drawColor, circleThickness)
                        cv2.line(img, (x1_start, y1_start), (x1, y1), drawColor, circleThickness)
                        cv2.line(imgCanvas, (x1_start, y1_start), (x1, y1), drawColor, circleThickness)
                else:
                    dragging = False

                # Take screenshot if three fingers are up
                if fingers[1] and fingers[2] and fingers[3] and not fingers[0] and not fingers[4]:
                    filename = f"screenshot_{uuid.uuid4().hex}.png"
                    save_path = os.path.join(screenshot_folder, filename)
                    cv2.imwrite(save_path, img)
                    print(f"Screenshot saved to {save_path}")

                # Erase all drawings if four fingers are up
                if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

                # Selection mode: Two fingers are up
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
                    if y1 < 125:
                        if 150 < x1 < 260:
                            header = overlayList[0]
                            drawColor = (0, 0, 255)
                        elif 360 < x1 < 470:
                            header = overlayList[1]
                            drawColor = (255, 255, 0)
                        elif 570 < x1 < 680:
                            header = overlayList[2]
                            drawColor = (0, 255, 0)
                        elif 790 < x1 < 910:
                            header = overlayList[3]
                            drawColor = (0, 0, 0)
                        elif 1010 < x1 < 1090:
                            header = overlayList[4]
                            brushThickness += 2
                            print(f"Brush thickness increased to {brushThickness}")
                        elif 1140 < x1 < 1220:
                            header = overlayList[5]
                            brushThickness = max(5, brushThickness - 2)  # Ensure brush thickness doesn't go below 5
                            print(f"Brush thickness decreased to {brushThickness}")

                # Drawing mode: Index finger is up
                if fingers[1] and not fingers[2]:
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    if drawColor == (0, 0, 0):
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    else:
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    xp, yp = x1, y1

            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgCanvas)

            img[0:125, 0:1280] = header

            # Draw laser pointer dot (laser_x, laser_y) without rendering on canvas
            cv2.circle(img, (laser_x, laser_y), laser_radius, (0, 255, 255), cv2.FILLED)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for painter page
@app.route('/painter')
def painter():
    return render_template('painter.html')

# Route for serving header images
@app.route('/header/<filename>')
def header_image(filename):
    return send_from_directory('header', filename)

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for meeting page
@app.route('/meeting')
def meeting():
    return render_template('meeting.html')

# Route for viewing screenshots
@app.route('/screenshots')
def screenshots():
    images = os.listdir(screenshot_folder)
    return render_template('screenshots.html', images=images)

# Route for serving screenshots
@app.route('/screenshots/<filename>')
def get_screenshot(filename):
    return send_from_directory(screenshot_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)