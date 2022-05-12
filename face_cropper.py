import matplotlib.pyplot as plt
import glob, cv2, os
from deepface.detectors import FaceDetector
#              0        1      2        3          4             5
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
backend = backends[5]

# Create a face detector
face_detector = FaceDetector.build_model(backend)

# Face normalization (to use in preprocessing steps)
print(os.getcwd())
paths=glob.glob("./faces/*.jpg")

if not(os.path.exists("./faces_db/")):
	os.mkdir("./faces_db/")

i=0; j=0; list_of_nofaces = []
for path in paths:
	if not(os.path.exists("./faces_db/"+path)):
		try:
			i+=1
			filename = os.path.basename(path)
			img = cv2.imread(path)
			faces = FaceDetector.detect_faces(face_detector, backend, img, align = True)

			for face, (x, y, w, h) in faces:
				if w<50:
					print("Face removed on " + filename)
					continue # Remove small faces
				face = img[int(y):int(y+h), int(x):int(x+w)]
				face = cv2.normalize(face, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
				face = cv2.resize(face, (224,224))				

				if len(faces)>1:
					cv2.imwrite('./faces_db/' + str(j) + filename , face)
				else:
					cv2.imwrite('./faces_db/' + filename, face)

				j+=1

		except ValueError as e:
			print("Error: "+str(e))
			print(str(j) + " faces already found of" + str(i) + "images processed")
			list_of_nofaces.append(filename)
			continue

print('This are the images that have no faces: ')
print(list_of_nofaces)