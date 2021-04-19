import cv2 as cv

# Read image from your local file system
#original_image = cv.imread(r'C:\Users\kai\PycharmProjects\ViolaJonesAlgorithm\pexels-victoria-borodinova-1648387.jpg')

#Capture Camera
cap = cv.VideoCapture(0)

while cap.isOpened():
    #Capture Frame by Frame
    _, frame = cap.read()

    # Convert color image to grayscale
    grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Load the classifier and create a cascade object for face detection
    face_cascade = cv.CascadeClassifier(r"C:\Users\kai\PycharmProjects\libs+packages\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")

    #Use the Method detectMultiScale() of the face.cascade object to  look at subregions of the image in multiple scales,
    #to detect faces of various sizes
    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    
    #draw bounding Box
    for (column, row, width, height) in detected_faces:
        cv.rectangle(
            frame,
            (column, row),
            (column + width, row + height),
            (0, 255, 0),
            2
        )
    
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
