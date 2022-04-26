from Inference.Facenet import Facenet
import cv2

fn = Facenet()
camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    if ret:
        try:
            people_list = fn.Get_People_Identity_SVM(img)
            for identity, distance, _, box in people_list:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                print("Identity : ", identity, " - ", "Distance : ", distance)
            print("---------------------------------------------------")
        except:
            print("No face")

        cv2.imshow("Picture", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# 255,0,0 : blue
