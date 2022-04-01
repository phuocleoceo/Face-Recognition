from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from os.path import join, curdir
from sklearn.svm import SVC
import joblib
import json


class SVM:
    def __init__(self):
        self.classifier_path = join(curdir, "SVM_Model", "svm.sav")
        # Load database chứa các vector đặc trưng
        db_path = join(curdir, "Database", "Database.json")
        with open(db_path, "r") as db:
            self.database = json.load(db)

    def train(self):
        """
        Hàm train model SVM
        """
        # Từ dict database, ta biến thành 2 list chữa label và embd tương ứng
        face_labels = []
        face_embeddings = []
        for label, embedding in self.database.items():
            for embd in embedding:
                face_labels.append(label)
                face_embeddings.append(embd)

        # Phân chia tập train và test
        # Từ 1 embd ta dự đoán ra label tương ứng
        X_train, X_test, Y_train, Y_test = train_test_split(face_embeddings, face_labels,
                                                            test_size=0.2, random_state=42)

        # SVC classifier
        model = SVC(kernel="linear", probability=True)

        # Fit model
        model.fit(X_train, Y_train)

        # Đánh giá kết quả
        print(">> SVM training accuracy :", accuracy_score(Y_train, model.predict(X_train))*100)
        print(">> SVM testing accuracy :", accuracy_score(Y_test, model.predict(X_test))*100)

        # Lưu lại model đẻ sử dụng sau này
        with open(self.classifier_path, "wb") as f:
            joblib.dump(model, f)

    def load_model(self):
        with open(self.classifier_path, "rb") as f:
            model = joblib.load(f)
