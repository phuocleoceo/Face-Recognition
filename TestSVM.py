from Classifier.SVM import SVM
from Face.Facenet import Facenet
import cv2

# svm = SVM()
# svm.train()

img = cv2.imread("./Dataset/Test/ngan.jpg")

fn = Facenet()
identity, embd = fn.Get_People_Identity_SVM(img)[0]
print(embd)
print(identity)
