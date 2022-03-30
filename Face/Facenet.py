from Face.FaceRecognition import FaceRecognition
from Face.FaceDetection import FaceDetection


class Facenet:
    def __init__(self):
        self.detector = FaceDetection()
        self.recognizer = FaceRecognition()

    def Get_People_Identity(self, image, resize=True, scale=4):
        """
        Hàm trả về tên khuôn mặt
        image : ảnh
        resize : có giảm kích thước ảnh hay không nhằm tăng tốc độ xử lý
        scale : tỉ lệ giảm kích thước ảnh
        """
        # Detect hình chữ nhật bao quanh khuôn mặt
        rec = self.detector.Detect(image, resize, scale)
        # Ma trận gương mặt được detect
        face_crop = self.detector.Crop_Face(image, rec)

        # Nhận diện nhiều gương mặt cho chắc
        identity = []
        for face in face_crop:
            # Trích xuất đặc trưng
            face_embd = self.recognizer.Get_Face_Embedding(face)
            # Nhận dạng
            person_name, distance = self.recognizer.Face_Identify(face_embd)
            identity.append((person_name, distance))
        return identity
