import cv2
import numpy as np
import time
import math

# ==============================
# CONFIGURATION
# ==============================

BOX_WIDTH = 200
BOX_HEIGHT = 200

WARNING_DIST = 120
DANGER_DIST = 40

MIN_HAND_AREA = 3000


# ==============================
# FUNCTIONS
# ==============================

def segment_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color mask (two hue ranges for safety)
    lower1 = np.array([0, 20, 70], np.uint8)
    upper1 = np.array([20, 255, 255], np.uint8)

    lower2 = np.array([160, 20, 70], np.uint8)
    upper2 = np.array([179, 255, 255], np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = mask1 | mask2

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)

    return mask


def find_hand_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < MIN_HAND_AREA:
        return None

    return contour


def find_fingertip(contour):
    if contour is None:
        return None

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    hull = cv2.convexHull(contour)
    if hull is None or len(hull) == 0:
        return None

    max_dist = -1
    fingertip = None

    for point in hull:
        x, y = point[0]
        dist = (x - cx) ** 2 + (y - cy) ** 2
        if dist > max_dist:
            max_dist = dist
            fingertip = (int(x), int(y))

    return fingertip


def distance_to_box(px, py, rect):
    x1, y1, x2, y2 = rect

    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)

    return math.sqrt(dx*dx + dy*dy)


def classify_distance(dist):
    if dist <= DANGER_DIST:
        return "DANGER", (0, 0, 255)
    elif dist <= WARNING_DIST:
        return "WARNING", (0, 255, 255)
    else:
        return "SAFE", (0, 255, 0)


# ==============================
# MAIN PROGRAM
# ==============================

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera Error!")
        return

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape

        box = (w//2 - BOX_WIDTH//2, h//2 - BOX_HEIGHT//2,
               w//2 + BOX_WIDTH//2, h//2 + BOX_HEIGHT//2)

        mask = segment_hand(frame.copy())
        contour = find_hand_contour(mask)
        fingertip = find_fingertip(contour)

        state = "NO HAND"
        color = (255, 255, 255)

        if fingertip:
            fx, fy = fingertip
            cv2.circle(frame, (fx, fy), 10, (255, 0, 0), -1)

            dist = distance_to_box(fx, fy, box)
            state, color = classify_distance(dist)

            cv2.putText(frame, f"Dist: {int(dist)}px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
        cv2.putText(frame, f"STATE: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        if state == "DANGER":
            cv2.putText(frame, "DANGER DANGER", (150, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 4)

        # FPS display
        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Hand Danger Detection", frame)
        cv2.imshow("Mask (debug)", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
