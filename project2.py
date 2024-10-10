import cv2 as cv
import imutils
import os

# Load the Haar Cascade Classifiers
face_cascade_path = os.path.join("C:/Users/ASUS/Desktop/Bhavesh python", "haarcascade_frontalface_default.xml")
eye_cascade_path = os.path.join("C:/Users/ASUS/Desktop/Bhavesh python", "haarcascade_eye.xml")

face_cascade = cv.CascadeClassifier(face_cascade_path)
eye_cascade = cv.CascadeClassifier(eye_cascade_path)

# Check if the cascades were loaded successfully
if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load Haar Cascade files.")
    exit()

# Read the Image
img_path = "File path"
img = cv.imread(img_path)

if img is None:
    print("Error: Image not found.")
    exit()

# Resize the Image
img_r = imutils.resize(img, width=1000)

# Convert Image to Grayscale
gray = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

# Detect Faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# Print the number of detected faces
print(f"Faces detected: {len(faces)}")

# If no faces are detected, print a message
if len(faces) == 0:
    print("No faces detected.")
else:
    print("Face detected.")

# Process each face
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv.rectangle(img_r, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    # Define the region of interest (ROI) for eyes within the detected face
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img_r[y:y + h, x:x + w]
    
    # Detect Eyes within the ROI
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
    
    # Draw rectangles around detected eyes
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), 2)

# Display the Result
cv.imshow('Detected Faces and Eyes', img_r)
cv.waitKey(0)
cv.destroyAllWindows()
