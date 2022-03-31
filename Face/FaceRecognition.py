# from Face.inception_resnet_v1 import InceptionResNetV1
from keras.models import load_model
from os.path import join, curdir
import numpy as np
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

    def Preprocessing_IMG(self, image):
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

    def L2_Normalize(self, embed, axis=-1, epsilon=1e-10):
        """
        Áp dụng chuẩn hóa Norm2 để tránh Overfitting
        embed : vector đặc trưng được model trích xuất
        """
        square_sum = np.sum(np.square(embed), axis=axis, keepdims=True)
        square_sum = np.maximum(square_sum, epsilon)
        return embed / np.sqrt(square_sum)

    def Get_Face_Embedding(self, face):
        # Tiền xử lý ảnh khuôn mặt sau đó thêm chiều
        processed_face = self.Preprocessing_IMG(face)
        processed_face = np.expand_dims(processed_face, axis=0)

        # Dùng model để trích xuất vector đặc trưng
        face_embedding = self.model.predict(processed_face)
        # Chuẩn hóa Norm2
        face_embedding = self.L2_Normalize(face_embedding)
        return face_embedding
