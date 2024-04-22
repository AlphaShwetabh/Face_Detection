import threading
import cv2
import numpy as np

from deepface import DeepFace

models=["Facenet","DeepID","DeepFace","OpenFace","Facenet512","VGG-Face","Dlib","SFace","ArcFace"]
#DeepID Facenet512 OpenFace
obj=DeepFace.verify(img1_path = "shubha.jpg",img2_path = "shubha/s1.jpeg", model_name = models[1])
print(1-obj['distance'])

obj=DeepFace.verify(img1_path = "shubha.jpg",img2_path = "shubha/s2.jpeg", model_name = models[1])
print(1-obj['distance'])

obj=DeepFace.verify(img1_path = "shubha.jpg",img2_path = "shubha/s3.jpeg", model_name = models[1])
print(1-obj['distance'])

obj=DeepFace.verify(img1_path = "shubha.jpg",img2_path = "shubha/s4.jpeg", model_name = models[1])
print(1-obj['distance'])

obj=DeepFace.verify(img1_path = "shubha.jpg",img2_path = "shubha/s5.jpeg", model_name = models[1])
print(1-obj['distance'])

obj=DeepFace.verify(img1_path = "shubha.jpg",img2_path = "shubha/s6.jpeg", model_name = models[1])
print(1-obj['distance'])

obj=DeepFace.verify(img1_path = "shubha.jpg",img2_path = "shubha/s7.jpeg", model_name = models[1])
print(1-obj['distance'])

obj=DeepFace.verify(img1_path = "shubha.jpg",img2_path = "shubha/s8.jpeg", model_name = models[1])
print(1-obj['distance'])

obj=DeepFace.verify(img1_path = "shubha.jpg",img2_path = "shubha/s9.jpeg", model_name = models[1])
print(1-obj['distance'])

obj=DeepFace.verify(img1_path = "shubha.jpg",img2_path = "shubha/s10.jpeg", model_name = models[1])
print(1-obj['distance'])

#fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#axs [0].imshow(plt. imread (' face-db/shwetabh/s1.png'))
#axs [1].imshow(plt. imread ('face-db/shwetabh/s1.png'))

cv2.destroyAllWindows() 