from Face.FaceRecognition import FaceRecognition
from os.path import join
from os import listdir
import json
import cv2

People_img_path = "Dataset/People"
DB_path = "Database"
DB_file = "Database.json"


def Check_Database():
    """
    Hàm kiểm tra xem đã có file Database chưa
    """
    db_file = listdir(DB_path)
    # Nếu chưa có thì tạo thôi
    if DB_file not in db_file:
        # x : nếu tệp đã tồn tại thì không mở được
        with open(join(DB_path, DB_file), "x") as db:
            json.dump({}, db)


def Save_Feature_To_Database(person_name, feature_vector):
    """
    Hàm lưu vector đặc trưng của 1 người vào Database
    person_name : Tên ngươi
    feature_vector : vector đặc trưng tương ứng
    """
    # Load database
    with open(join(DB_path, DB_file), "r") as db:
        data = json.load(db)

    # Thêm trường mới
    data[person_name] = feature_vector

    # Cập nhật lại database
    with open(join(DB_path, DB_file), "w") as db:
        json.dump(data, db)
        print(f"{person_name} feature is added to database !")


def Get_People_Feature():
    recog = FaceRecognition()

    people = listdir(People_img_path)
    for person in people:
        name = person.split('.')[0]
        face = cv2.imread(join(People_img_path, person))
        face_embd = recog.get_face_embedding(face).flatten().tolist()
        Save_Feature_To_Database(name, face_embd)


if __name__ == "__main__":
    Check_Database()
    Get_People_Feature()