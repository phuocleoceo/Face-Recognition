from Face.Facenet import Facenet
import cv2

img = cv2.imread("./Dataset/Test/thinh.jpg")

fn = Facenet()

identity = fn.Get_People_Identity(img)
print(identity)
