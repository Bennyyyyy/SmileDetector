import cv2

# Face Classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Grab Webcam Feed

webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read() # This will read one single frame but in the while loop it will read many frames
    if not successful_frame_read:
        break
     
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 200, 50), 4)


    cv2.imshow('Smile Detector', frame)
    cv2.waitKey(1)

# Cleanup of resources such as webcam
webcam.release() 
# Closes all windows so everything is gone
cv2.destroyAllWindows()
