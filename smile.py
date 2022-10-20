import cv2

# Face Classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Grab Webcam Feed

webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read() # This will read one single frame but in the while loop it will read many frames
    if not successful_frame_read:
        break
     
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_grayscale, minNeighbors=10)
    # Scale factor --- Blurs image to get rid of false positives but still maintain the smile and face
    # minNeighbors --- There must be 20 neighboring rectangles for it to count as a smile

    for (x, y, w, h) in faces:
        
        # Draws a rectangle aroud the face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 200, 50), 4)

        # Get the sub-array meaning the square around the face
        # This uses array slicing to slice just the face out of the frame
        # Improves efficiency and accuracy
        the_face = frame[y:y+h, x:x+w]
        
        #This converts 
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # This checks only for the frame around the face for a smile
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20) 

        #This takes the x,y,h,w in the smiles array and draws a square on what it thinks is a smile
        # for(x_, y_, w_, h_) in smiles:
        #     cv2.rectangle(the_face, (x_,y_), (x_+w_, y_+h_), (50, 50, 200), 4)

        #Checks if the smiles array is greater than one meaning there is a smile
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
    
    
    cv2.imshow('Smile Detector', frame)
    cv2.waitKey(1)

# Cleanup of resources such as webcam
webcam.release() 
# Closes all windows so everything is gone
cv2.destroyAllWindows()
