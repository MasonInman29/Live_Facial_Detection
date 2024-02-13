**Facial Recognition with OpenCV**

There are two parts to this project, one part for static images and one for videos. I completed the static image facial detection before moving to the live video detection. The process is relatively the same.

OpenCV does a lot of the dirty work here, but this is the basic process:
1. Gain access to file/picture/video feed
2. If using a video feed, just treat each frame as its own video
3. Grayscale the image for easier image processing
4. Use one of OpenCV's many classifiers (I used haarcascades) to detect faces of the image/frame

Then I draw rectangles around every face and keep tract of how many faces are being detected as well.
