import os
import cv2

#print(os.listdir())

face_cascade = cv2.CascadeClassifier('Chapter07/haarcascade_frontalface_default.xml')

def detect_faces(img, draw_box=True):
	# 이미지를 흑백으로 바꾼다.
	grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# 얼굴을 검출한다.
	faces = face_cascade.detectMultiScale(grayscale_img, scaleFactor=1.1,
		minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

	face_box, face_coords = None, []

	for (x, y, w, h) in faces:
		if draw_box:
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
		face_box = img[y:y+h, x:x+w]
		face_coords = [x,y,w,h]

	return img, face_box, face_coords

if __name__ == "__main__":
	files = os.listdir('Chapter07/sample_faces')
	images = [file for file in files if 'jpg' in file]
	for image in images:
		img = cv2.imread('Chapter07/sample_faces/' + image)
		detected_faces, _, _ = detect_faces(img)
		cv2.imwrite('Chapter07/sample_faces/detected_faces/' + image, detected_faces)