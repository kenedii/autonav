# This code is meant for testing camera indexes in OpenCV to view the output. 
# Use camera_cv2.py, realsense_cv2.py, or realsense_pyrealsense.py instead for functions to easily interact with the camera.
import cv2

cap =  cv2.VideoCapture(1)

if not cap.isOpened():
	print("Error.")
	exit()
print("Camera opened. Press SPACE to take a photo, ESC to quit.")
while True:
	ret, frame = cap.read()
	if not ret:
		print("Error.")
		break
	cv2.imshow('Camera - Press SPACE to capture', frame)
	key = cv2.waitKey(1) & 0xFF

	# Space = take photo
	if key == 32: #ascii for space
		filename = "camera_1test.png"
		cv2.imwrite(filename, frame)
		print("Photo saved as " + filename )
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()

