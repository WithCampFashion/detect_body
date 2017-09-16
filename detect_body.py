import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def auto_scan_image_via_webcam():
    try:
        cap = cv2.VideoCapture(0)
    except:
        print 'cannot load camera!'
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print 'cannot load camera!'
            break

        k = cv2.waitKey(10)
        if k == 27:
            break

        gray = cv2.GaussianBlur(frame, (3, 3), 0)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=3,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            cv2.imshow("WebCam", frame)
            continue
        else:
            roi_gray = frame[faces[0][1] - 50:faces[0][1] + faces[0][3] + 50, faces[0][0] - 50:faces[0][0] + faces[0][2] + 50]
            x = faces[0][0] - 50
            y = faces[0][1] - 50
            w = faces[0][2] + 50
            h = faces[0][3] + 50
            cv2.imwrite('dataset/face_detected.jpg', roi_gray)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("WebCam", frame)

            continue

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    auto_scan_image_via_webcam()