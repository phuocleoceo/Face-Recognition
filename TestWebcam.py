from Inference.Facenet import Facenet
import cv2

fn = Facenet()
camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    if ret:
        cv2.imshow("Picture", img)

        try:
            identity, distance, _ = fn.Get_People_Identity_SVM(img)[0]
            print("Identity : ", identity, " - ", "Distance : ", distance)
        except:
            print("No face")

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# 255,0,0 : blue
