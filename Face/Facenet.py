from Face.FaceRecognition import FaceRecognition
from Face.FaceDetection import FaceDetection
from scipy.spatial import distance
from os.path import join, curdir
from Classifier.SVM import SVM
import json


class Facenet:
    def __init__(self):
        self.detector = FaceDetection()
        self.recognizer = FaceRecognition()
        self.classifier = SVM()
        # Load database chứa các vector đặc trưng
        db_path = join(curdir, "Database", "Database.json")
        with open(db_path, "r") as db:
            self.database = json.load(db)

    def Euclidean_Distance(self, embd_db, embd_recog):
        """
        Hàm tính khoảng cách Euclidean giữa 2 vector
        embd_db : vector khuôn mặt đang được lưu trong database
        embd_recog : vector khuôn mặt đang nhận dạng
        """
        # return np.sqrt(np.sum((embd_db-embd_recog)**2))
        return distance.euclidean(embd_db, embd_recog)

    def Face_Identify(self, face_embedding):
        """
        Hàm xác định danh tính khuôn mặt
        face_embedding : vector đặc trưng khuôn mặt đang xét
        """
        # Từ điển chứa khoảng cách Euclidean từ face_embedding đến các vector trong database
        distance = {}
        for name, embd in self.database.items():
            # Mỗi người sẽ có nhiều vector đặc trưng
            # Ta tính khoảng cách từ mỗi vector đặc trưng đó đến face_embedding
            euc_dist = [self.Euclidean_Distance(ed, face_embedding) for ed in embd]
            # Gán distance của 1 người thành min của mảng khoảng cách vừa tính được
            distance[name] = min(euc_dist)

        # Lấy key có value nhỏ nhất trong distance
        person_name = min(distance, key=distance.get)
        min_dist = distance[person_name]

        # Nếu khoảng cách > 1 thì người đang xét không có trong Database
        if min_dist > 1:
            person_name = "UNKNOWN"
        return person_name, min_dist

    def Get_People_Identity(self, image, resize=True, scale=4):
        """
        Hàm trả về tên khuôn mặt
        image : ảnh
        resize : có giảm kích thước ảnh hay không nhằm tăng tốc độ xử lý
        scale : tỉ lệ giảm kích thước ảnh
        """
        # Detect hình chữ nhật bao quanh khuôn mặt
        rec = self.detector.Detect_Face(image, resize, scale)
        # Ma trận gương mặt được detect
        face_crop = self.detector.Crop_Face(image, rec)

        # Nhận diện nhiều gương mặt cho chắc
        identity = []
        for face in face_crop:
            # Trích xuất đặc trưng
            face_embd = self.recognizer.Get_Face_Embedding(face)
            # Nhận dạng
            person_name, distance = self.Face_Identify(face_embd)
            identity.append((person_name, distance, face_embd))
        return identity

    def Get_People_Identity_SVM(self, image, resize=True, scale=4):
        """
        Hàm trả về tên khuôn mặt sử dụng SVM
        image : ảnh
        resize : có giảm kích thước ảnh hay không nhằm tăng tốc độ xử lý
        scale : tỉ lệ giảm kích thước ảnh
        """
        svm = self.classifier.load_model()
        # Detect hình chữ nhật bao quanh khuôn mặt
        rec = self.detector.Detect_Face(image, resize, scale)
        # Ma trận gương mặt được detect
        face_crop = self.detector.Crop_Face(image, rec)

        # Nhận diện nhiều gương mặt cho chắc
        identity = []
        for face in face_crop:
            # Trích xuất đặc trưng
            face_embd = self.recognizer.Get_Face_Embedding(face)
            # Nhận dạng
            person_name = svm.predict(face_embd)[0]
            identity.append((person_name, face_embd))
        return identity
