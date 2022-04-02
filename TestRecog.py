from Face.Facenet import Facenet
import matplotlib.pyplot as plt
import cv2


# img = cv2.imread("./Dataset/Test/thinh.jpg")
img = cv2.imread("./Dataset/Test/tzuyu.png")

fn = Facenet()

identity, distance, embd = fn.Get_People_Identity(img)[0]
print("Who is this ? => ", identity)
print("Euclidean Distance : ", distance)

plt.plot(embd[0])
plt.show()
