from Face.Facenet import Facenet
import matplotlib.pyplot as plt
from Classifier.SVM import SVM
import cv2

# svm = SVM()
# svm.train()

img = cv2.imread("./Dataset/Test/thinh.jpg")

fn = Facenet()
identity, embd = fn.Get_People_Identity_SVM(img)[0]
print("Who is this ? => ", identity)

plt.plot(embd[0])
plt.show()
