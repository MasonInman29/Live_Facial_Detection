import cv2


def detect_bounding_box(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Face detection
        #detectMultiScale for faces of different sizes
        #scaleFactor = 10% image size reduction
        #minNeighbors prevents false positives, eg.1 or 2 will have many false positives
        #minSize - minmimum size of a face to return
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(30, 30))
    i = 1
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(frame, 'Face: ' + str(i), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0), 2)
        i += 1
        
    return faces


#PRE TRAINED face classifier - not great unless face is straight on
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 0 will go to system default camera
video_capture = cv2.VideoCapture(0)

while True:
    result, video_frame = video_capture.read()  #get frame
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(video_frame)

    cv2.imshow("Live Facial Detection", video_frame) # display the processed frame

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()