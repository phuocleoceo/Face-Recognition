from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
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

        # Phân chia tập train và test
        # Từ 1 embd ta dự đoán ra label tương ứng
        X_train, X_test, Y_train, Y_test = train_test_split(face_embeddings, face_labels,
                                                            test_size=0.2, random_state=42)

        # SVC classifier
        model = SVC(kernel="linear", probability=True)

        # Fit model
        model.fit(X_train, Y_train)

        # Đánh giá kết quả
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        self.Visualize(Y_train, Y_train_pred, Y_test, Y_test_pred)

        # Lưu lại model đẻ sử dụng sau này
        with open(self.classifier_path, "wb") as f:
            joblib.dump(model, f)

    def load_model(self):
        with open(self.classifier_path, "rb") as f:
            model = joblib.load(f)
        return model

    def Visualize(self, Y_train, Y_train_pred, Y_test, Y_test_pred):
        Y_train_accuracy = accuracy_score(Y_train, Y_train_pred)*100
        Y_test_accuracy = accuracy_score(Y_test, Y_test_pred)*100
        print(">> SVM training accuracy :", Y_train_accuracy)
        print(">> SVM testing accuracy :", Y_test_accuracy)

        plt.figure(1, figsize=(6, 4))
        plt.title("Train data with accuracy "+str(Y_train_accuracy)+" (%)")
        sns.heatmap(confusion_matrix(Y_train, Y_train_pred), cmap="YlGnBu", annot=True, fmt='g')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig(self.visualization_path+"/train_confusion.png")

        plt.figure(2, figsize=(6, 4))
        plt.title("Test data with accuracy "+str(Y_test_accuracy)+" (%)")
        sns.heatmap(confusion_matrix(Y_test, Y_test_pred), cmap="YlGnBu", annot=True, fmt='g')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig(self.visualization_path+"/test_confusion.png")
