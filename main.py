import cv2 as cv

# Read image from your local file system
original_image = cv.imread(r'C:\Users\kai\PycharmProjects\ViolaJonesAlgorithm\pexels-victoria-borodinova-1648387.jpg')

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

# Load the classifier and create a cascade object for face detection
face_cascade = cv.CascadeClassifier(r"C:\Users\kai\PycharmProjects\libs+packages\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")

#Use the Method detectMultiScale() of the face.cascade object to  looks at subregions of the image in multiple scales,
#to detect faceses of various sizes

detected_faces = face_cascade.detectMultiScale(grayscale_image)

for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )

cv.imshow('Image', original_image)
cv.waitKey(0)
cv.destroyAllWindows()


#cv.imshow("test", face_cascade)
#cv.waitKey(0)