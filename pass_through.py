import pyvirtualcam
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN

#from yolov5 import YOLOv5





cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()


#detector = MTCNN()

#def filter(frame):
#    faces = detector.detect_faces(frame)
#    for face in faces:
#       x1, y1, width, height = face['box']
#        x2, y2 = x1 + width, y1 + height
#        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

# set model params
model_path = "yolov5/weights/yolov5s.pt" # it automatically downloads yolov5s model to given path

device = "cpu" # "cuda" or "cpu"

# init yolov5 model
#yolov5 = YOLOv5(model_path, device)



#    return frame

def filter2(frame):
    results = yolov5.predict(frame)
    
    results.render()
    
    return results.imgs[0]


# Load the cascade
#face_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def filter3(frame):
   
    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        #r = int((w+h)/4)
        #r1 = int(r / 2)
        #center = ( x + r , y + r)
        #cv2.circle(frame, center, r, (255, 255, 255), -1)
        #cv2.circle(frame, center, r1, (0, 0, 0), -1)

        center = ( x + int(w/2) , y + int(h/2))

        cv2.ellipse(frame, center, (int(w/2),int(h/2)), 0, 0, 360, (255, 255, 255), -1)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame




overlay = cv2.imread('PngItem_1330705.png')


with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
    print(f'Using virtual camera: {cam.device}')

    #frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB

    while True:
        # read the image
        _, frame = cap.read()

 
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        img = filter3(img)


        #added_image = cv2.addWeighted(img,0.4,overlay,0.1,0)
        #fgbg.apply(img);

        # Get background
        #background = fgbg.getBackgroundImage()


        #fgmask = fgbg.apply(frame)

        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        #pixel_values = image.reshape((-1, 3))
        #frame[:] = cam.frames_sent % 255  # grayscale animation
        cam.send(img)
        cam.sleep_until_next_frame()
