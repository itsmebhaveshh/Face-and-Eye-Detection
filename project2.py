import cv2
import imutils
import time

# Load Haar cascade files
face_cascade = cv2.CascadeClassifier("Enter the Xml file path", "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("Enter the Xml file path", "haarcascade_eye.xml")

# Start webcam
vid = cv2.VideoCapture(0)  # 0 indicates the primary webcam

# Timing variables
start_time = None  # Initialize start time
no_face_duration = 20  # Duration to keep the frame open if no faces are detected (in seconds)
face_detected_duration = 3  # Duration to keep the frame open if a face is detected (in seconds)

# Flags to ensure output is printed only once
face_message_printed = False
no_face_message_printed = False

while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break  # Exit if the frame is not captured

    # Resize the frame for consistency
    img_r = imutils.resize(frame, width=1000)
    
    # Convert to grayscale for Haar cascade
    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if start_time is None:
        start_time = time.time()  # Record the start time when the frame opens

    # Adjust the duration based on face detection
    elapsed_time = time.time() - start_time
    if len(faces) > 0:
        duration = face_detected_duration
        if not face_message_printed:
            print(f"Number of faces detected: {len(faces)}")
            face_message_printed = True
    else:
        duration = no_face_duration
        if not no_face_message_printed and elapsed_time >= duration:
            print("Number of faces detected: 0")
            no_face_message_printed = True

    # Break the loop if the frame has been displayed for the specified duration
    if elapsed_time >= duration:
        break

    # Draw rectangles and detect eyes if faces are present
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img_r, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Extract regions of interest (ROI) for face and eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img_r[y:y + h, x:x + w]
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    # Display the processed video feed
    cv2.imshow('Frame', img_r)

    # Exit on pressing 'q' or 'Esc'
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or k == 27:
        print("User terminated the program.")
        break

# Release the webcam and close all windows
vid.release()
cv2.destroyAllWindows()
