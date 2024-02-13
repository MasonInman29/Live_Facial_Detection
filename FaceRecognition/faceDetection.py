import cv2


def detect_bounding_box(frame, option):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Face detection
        #detectMultiScale for faces of different sizes
        #scaleFactor = 10% image size reduction
        #minNeighbors prevents false positives, eg.1 or 2 will have many false positives
        #minSize - minmimum size of a face to return
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    numFaces = 1
    for (x, y, w, h) in faces:
        #testing frame
        if (option == 1): #green rectangles
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        elif (option == 2): #smiley face
            radius = int(((h+w) / 2) / 1.9)
            #yellow face
            cv2.circle(frame, (x + int(w/2), y + int(h/2)), radius, (50, 200, 255), -1)
            #eyes
            cv2.circle(frame, (x + int(w * .25), y + int(h * .33)), int(radius / 4), (50, 50, 50), -1)
            cv2.circle(frame, (x + int(w * .75), y + int(h * .33)), int(radius / 4), (50, 50, 50), -1)

            # Mouth
            center_x = x + int(w / 2)
            center_y = y + int(h * 0.6)
            axes_length = int(w * 0.25)
            start_angle = 0
            end_angle = 180
            color = (50, 50, 50) 
            thickness = 5
            cv2.ellipse(frame, (center_x, center_y), (axes_length, int(axes_length * 0.5)), 0, start_angle, end_angle, color, thickness)
        
        #small text in top left of face
        cv2.putText(frame, 'Face: ' + str(numFaces), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)
        numFaces += 1
        
    #text in top left of screen
    cv2.putText(frame, "Total Faces: " + str(numFaces-1), (20,20), cv2.FONT_HERSHEY_PLAIN,1.1,(0,255,0),2)
    cv2.putText(frame, "Option: " + str(option), (20,40), cv2.FONT_HERSHEY_PLAIN,1.1,(0,255,0),2)
    return faces


#PRE TRAINED face classifier - not great unless face is straight on
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 0 will go to system default camera
video_capture = cv2.VideoCapture(0)

option = 1 #initialize value 
while True:
    result, video_frame = video_capture.read()  #get frame
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(video_frame, option)

    cv2.imshow("Live Facial Detection", video_frame) # display the processed frame

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"): #q to quit
        break
    elif key & 0xFF == ord("1"): #testing rectangles
        option = 1
    elif key & 0xFF == ord("2"): #smiley face
        option = 2

video_capture.release()
cv2.destroyAllWindows()