from mtcnn import MTCNN
import cv2


class FaceDetection():
    def __init__(self):
        # Sử dụng mạng MTCNN
        self.detector = MTCNN()

    def Detect_Face(self, image, resize=True, scale=4):
        """
        Hàm trả về tọa độ 2 đỉnh 2 góc của HCN bao quanh khuôn mặt
        image : ảnh
        resize : có giảm kích thước ảnh hay không nhằm tăng tốc độ xử lý
        scale : tỉ lệ giảm kích thước ảnh
        """
        img = image.copy()
        # Giảm kích thước
        if resize:
            width = img.shape[0]
            height = img.shape[1]
            img = cv2.resize(img, (height//scale, width//scale))

        # MTCNN phát hiện gương mặt
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_img)
        # Lưu lại tọa độ 2 đỉnh HCN
        rec = []
        for f in faces:
            x, y, w, h = f["box"]
            # Nếu có giảm kích thước thì toạ độ cần trả về giá trị trước khi resize
            if resize:
                rec.append([x*scale, y*scale, (x+w)*scale, (y+h)*scale])
            else:
                rec.append([x, y, x+w, y+h])
        return rec

    def Draw_Rec_To_Img(self, image, rec):
        """
        Hàm vẽ khung HCN quanh ảnh khuôn mặt
        image : ảnh
        rec : list chứa tọa độ HCN
        """
        img = image.copy()
        for x1, y1, x2, y2 in rec:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=5)
        return img

    def Crop_Face(self, image, rec):
        """
        Hàm trả về ma trận của khuôn mặt detect được
        image : ảnh
        rec : list chứa tọa độ HCN
        """
        img = image.copy()
        faces = []
        for x1, y1, x2, y2 in rec:
            faces.append(img[y1:y2, x1:x2])
        return faces
