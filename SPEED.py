import cv2
import numpy as np
import pygame

# Initialize Pygame mixer
pygame.mixer.init()

# Load the audio file
pygame.mixer.music.load(r"C:\Users\denos\OneDrive\Documents\Speed-Breaker-Detector-1\speed-source\slow down audio.mp3")


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines):
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                if abs(y2 - y1) < 20:  # Change the value as needed
                    pygame.mixer.music.play()
    except:
        pass


def lane_detection(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 50, 150)

    height = image.shape[0]
    width = image.shape[1]
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    result = cv2.addWeighted(image, 0.8, line_img, 1, 0)
    return result


cap = cv2.VideoCapture(r"C:\Users\denos\OneDrive\Documents\Speed-Breaker-Detector-1\speed-source\final_vid_input.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = lane_detection(frame)
    cv2.imshow("Lane Detection", result)

    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
