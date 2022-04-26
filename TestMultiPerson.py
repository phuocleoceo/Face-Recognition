from Inference.Facenet import Facenet
import cv2

fn = Facenet()
camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    if ret:
        try:
            # Lấy ra danh tính, khoảng cách và HCN bao gương mặt
            people_list = fn.Get_People_Identity_SVM(img)
            # Lặp qua tất cả các gương mặt phát hiện được
            for identity, distance, _, box in people_list:
                # Vẽ hình chữ nhật bao quanh gương mặt
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                # Viết danh tính lên hình
                img = cv2.putText(img, identity, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (0, 255, 0), 2, cv2.LINE_AA)
                # In ra console
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
