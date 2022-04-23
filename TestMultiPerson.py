from Inference.Facenet import Facenet
import cv2

fn = Facenet()
camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    if ret:
        cv2.imshow("Picture", img)

        try:
            people_list = fn.Get_People_Identity_SVM(img)
            for identity, _, _ in people_list:
                print("Identity : ", identity)
            print("----------------------------------------")
        except:
            print("No face")

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# 255,0,0 : blue
