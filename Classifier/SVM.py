from sklearn.metrics import accuracy_score, confusion_matrix
from os.path import join, curdir
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
import joblib
import json


class SVM:
    def __init__(self):
        self.classifier_path = join(curdir, "SVM_Model", "svm.sav")
        self.visualization_path = join(curdir, "SVM_Model")
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

        # SVC classifier
        model = SVC(kernel="linear", probability=True)

        # Fit model
        # Từ 1 embd ta dự đoán ra label tương ứng
        model.fit(face_embeddings, face_labels)

        # Đánh giá kết quả
        face_pred = model.predict(face_embeddings)
        self.Visualize(face_labels, face_pred)

        # Lưu lại model đẻ sử dụng sau này
        with open(self.classifier_path, "wb") as f:
            joblib.dump(model, f)

    def load_model(self):
        """
        Hàm load SVM model để sử dụng khi nhận dạng gương mặt
        """
        with open(self.classifier_path, "rb") as f:
            model = joblib.load(f)
        return model

    def Visualize(self, face_labels, face_pred):
        """
        Hàm trực quan hóa kết quả huấn luyện SVM
        """
        pred_accuracy = accuracy_score(face_labels, face_pred)*100
        print(">> SVM accuracy :", pred_accuracy)

        plt.figure(figsize=(6, 4))
        plt.title("Train data with accuracy "+str(pred_accuracy)+" (%)")
        sns.heatmap(confusion_matrix(face_labels, face_pred), cmap="YlGnBu", annot=True, fmt='g')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig(self.visualization_path+"/svm_confusion.png")
