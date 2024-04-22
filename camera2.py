import threading
import cv2
import random

from deepface import DeepFace

cap=cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter=0
cout=0

face_match=False

reference_img= cv2.imread("shubha.jpg")
reference_img2= cv2.imread("aku.jpg")

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img)['verified']:
          face_match=True
          print(DeepFace.verify(frame, reference_img, model_name = "DeepID")['distance'])

        elif DeepFace.verify(frame, reference_img2)['verified']:
          face_match=True
          print(DeepFace.verify(frame, reference_img2, model_name = "DeepID")['distance'])

        else:
            print(DeepFace.verify(frame, reference_img2, model_name = "DeepID")['distance'])
            face_match=False

    except ValueError:
        face_match= False

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
      
        counter += 1

        if face_match:
            cout=round(random.uniform(4,5),2)
            cv2.putText(frame, "Match  !", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        else:
            cout=0
            cv2.putText(frame, "NOT Match  !", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

        cv2.imshow("Video",frame)
  
    key=cv2.waitKey(1)

    if key == ord('q'):
        break
  
cv2.destroyAllWindows()
cap.release()