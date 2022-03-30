from Face.inception_resnet_v1 import InceptionResNetV1
from keras.models import load_model
from scipy.spatial import distance
from os.path import join, curdir
import numpy as np
import json
import cv2


class FaceRecognition():
    def __init__(self):
        # Load lại pretrain model để sử dụng
        try:
            # Join thư mục gọi class với Model/Facenet_Keras.h5
            model_path = join(curdir, 'Model', 'FaceNet_Keras.h5')
            self.model = load_model(model_path)
        except:
            print("Cannot find pretrain model")
        # Load database chứa các vector đặc trưng
        db_path = join(curdir, 'Database', 'Database.json')
        with open(db_path, "r") as db:
            self.database = json.load(db)

    def PreprocessingIMG(self, image):
        """
        Hàm tiền xử lý ảnh
        """
        # Giảm kích thước về đúng với input của Facenet
        img = cv2.resize(image, (160, 160))
        img = np.asarray(img, 'float32')

        # per_image_standardization
        # Mean 0 phương sai 1, giúp ảnh khi xử lý không gặp trường hợp Chia cho 0
        mean = np.mean(img, axis=(0, 1, 2), keepdims=True)
        std = np.std(img, axis=(0, 1, 2), keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(img.size))
        processed_img = (img-mean) / std_adj

        return processed_img

    # Áp dụng chuẩn hóa Norm2 để tránh Overfitting
    def L2Normalize(self, embed, axis=-1, epsilon=1e-10):
        square_sum = np.sum(np.square(embed), axis=axis, keepdims=True)
        square_sum = np.maximum(square_sum, epsilon)
        return embed / np.sqrt(square_sum)

    def GetFaceEmbedding(self, face):
        # Tiền xử lý ảnh khuôn mặt sau đó thêm chiều
        processed_face = self.PreprocessingIMG(face)
        processed_face = np.expand_dims(processed_face, axis=0)

        # Dùng model để trích xuất vector đặc trưng
        face_embedding = self.model.predict(processed_face)
        # Chuẩn hóa Norm2
        face_embedding = self.L2Normalize(face_embedding)
        return face_embedding

    def CalculateDistance(self, embd_real, embd_candidate):
        """
        Hàm tính khoảng cách Euclidean giữa 2 vector
        embd_real : vector khuôn mặt đang được lưu trong database
        embd_candidate : vector khuôn mặt đang nhận dạng
        """
        # return np.sqrt(np.sum((embd_real-embd_candidate)**2))
        return distance.euclidean(embd_real, embd_candidate)

    def Identify(self, face_embedding):
        """
        Hàm xác định danh tính khuôn mặt
        face_embedding : vector đặc trưng khuôn mặt đang xét
        """
        distance = {}
        min_dist = 9999
        person_name = ""
        for name, embedding in self.database.items():
            distance[name] = self.CalculateDistance(embedding, face_embedding)
            if distance[name] < min_dist:
                min_dist = distance[name]
                person_name = name
        if min_dist > 1:
            person_name = "UNKNOWN"
        return person_name, min_dist
