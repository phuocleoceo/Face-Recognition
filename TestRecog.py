from Face.Facenet import Facenet
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("./Dataset/Test/thanh.jpg")

fn = Facenet()

identity, distance, embd = fn.Get_People_Identity(img)[0]
print(embd)
print(identity)
print(distance)
