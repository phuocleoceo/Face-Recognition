from Face.FaceRecognition import FaceRecognition
from Face.FaceDetection import FaceDetection
from os.path import join
from os import listdir
import json
import cv2


class SaveFeatureVector:
    def __init__(self):
        self.People_img_path = "Dataset/People"
        self.DB_path = "Database"
        self.DB_file = "Database.json"
        self.detector = FaceDetection()
        self.recognizer = FaceRecognition()

    def Check_Database(self):
        """
        Hàm kiểm tra xem đã có file Database chưa
        """
        db_file = listdir(self.DB_path)
        # Nếu chưa có thì tạo thôi
        if self.DB_file not in db_file:
            # x : nếu tệp đã tồn tại thì không mở được
            with open(join(self.DB_path, self.DB_file), "x") as db:
                json.dump({}, db)

    def Save_Feature_To_Database(self, person_name, feature_vector):
        """
        Hàm lưu vector đặc trưng của 1 người vào Database
        person_name : Tên ngươi
        feature_vector : vector đặc trưng tương ứng
        """
        # Load database
        with open(join(self.DB_path, self.DB_file), "r") as db:
            data = json.load(db)

        # Thêm trường mới
        data[person_name] = feature_vector

        # Cập nhật lại database
        with open(join(self.DB_path, self.DB_file), "w") as db:
            json.dump(data, db)
            print(f"{person_name} feature vector is added to database !")

    def Get_People_Feature(self):
        """
        Hàm đọc hình ảnh người, trích xuất đặc trưng để lưu vào DB
        """
        people = listdir(self.People_img_path)
        for p in people:
            # Đọc file ảnh, lấy tên người
            name = p.split('.')[0]
            face = cv2.imread(join(self.People_img_path, p))
            # Detect khuôn mặt
            rec = self.detector.Detect_Face(face, speed_up=True, scale_factor=4)
            # Crop ra khuôn mặt đầu tiên
            face = self.detector.Crop_Face(face, rec)[0]
            # Lấy vector đặc trưng rồi lưu vào database
            face_embd = self.recognizer.Get_Face_Embedding(face).flatten().tolist()
            self.Save_Feature_To_Database(name, face_embd)


if __name__ == "__main__":
    sfv = SaveFeatureVector()
    sfv.Check_Database()
    sfv.Get_People_Feature()
