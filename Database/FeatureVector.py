from Face.FaceRecognition import FaceRecognition
from Face.FaceDetection import FaceDetection
from os.path import join, sep
from imutils import paths
from os import listdir
import json
import cv2


class FeatureVector:
    def __init__(self, People_img_path):
        self.People_img_path = People_img_path
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

        # Nếu trong db đã có người này thì mình thêm vector thôi
        # Còn không thì tạo mảng mới
        if person_name not in data:
            data[person_name] = [feature_vector]
        else:
            data[person_name].append(feature_vector)

        # Cập nhật lại database
        with open(join(self.DB_path, self.DB_file), "w") as db:
            json.dump(data, db)
            print(f"{person_name} feature vector is added to database !")

    def Load_People_Img(self):
        """
        Hàm đọc danh sách hình ảnh người kèm nhãn (tên người)
        """
        print(">> Loading people images ...")
        image_paths = list(paths.list_images(self.People_img_path))
        person_img = []
        person_name = []
        for ip in image_paths:
            image = cv2.imread(ip)
            person_img.append(image)
            # Split bằng "/" rồi lấy tên thư mục (-1 là tên file ảnh)
            lbl = ip.split(sep)[-2]
            person_name.append(lbl)
        return person_img, person_name

    def Get_People_Feature(self):
        """
        Hàm đọc hình ảnh người, trích xuất đặc trưng để lưu vào DB
        """
        person_img, person_name = self.Load_People_Img()
        for p_img, p_name in zip(person_img, person_name):
            # Detect khuôn mặt
            rec = self.detector.Detect_Face(p_img, resize=True, scale=4)
            # Crop ra khuôn mặt đầu tiên
            face = self.detector.Crop_Face(p_img, rec)[0]
            # Lấy vector đặc trưng rồi lưu vào database
            face_embd = self.recognizer.Get_Face_Embedding(face).flatten().tolist()
            self.Save_Feature_To_Database(p_name, face_embd)
