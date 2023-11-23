import cv2
from models.face_module import FaceModule


model_path = "./logs/train/runs/2023-11-20_14-56-15/checkpoints/epoch_009.ckpt"
model = FaceModule.load_from_checkpoint(model_path)
faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height


def process_frame(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # Predict with the model
    landmarks = model.predict_step(img, faces)  # predict on an image
    for face in landmarks:
        for i in range(68):
            cv2.circle(img, (int(face[i*2]*w+x), int(face[i*2+1]*h+y)), 1, (0, 0, 255), -1)

    return img

while True:
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    img = process_frame(img)
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()