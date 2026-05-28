# Face Recognition

Hệ thống nhận diện khuôn mặt sử dụng **MTCNN** (phát hiện khuôn mặt) + **FaceNet/Inception-ResNet V1** (trích xuất vector đặc trưng) + **SVM / Euclidean Distance** (phân loại danh tính). Dự án viết bằng Python, dùng Keras để load model tiền huấn luyện và OpenCV để xử lý ảnh, hỗ trợ nhận diện theo thời gian thực qua webcam.

---

## 1. Tài nguyên cần tải

### 1.1. Dataset
Tải dataset ảnh tại Google Drive:
https://drive.google.com/drive/folders/1wpkjOfiTdVksfjVSfzyeBeho4f60Mw1P?usp=sharing

Sau khi tải, giải nén các thư mục vào `Dataset/`:

```
Dataset/
├── People/        # Dữ liệu enroll — dùng để xây Database vector
│   ├── Phuoc/1.jpg ... N.jpg
│   ├── Hoang/1.jpg ... N.jpg
│   └── ...        (20 người, mỗi người N ảnh)
├── Test/          # Test set có nhãn — đánh giá accuracy (testSVM.ipynb)
│   ├── Phuoc/...  (cùng tên thư mục với People để so nhãn)
│   └── ...
├── UNKNOWN/       # Ảnh ngẫu nhiên không thuộc 20 người (vd: tzuyu.png) — demo trong API.ipynb
├── UNKNOWN-TEST/  # Test set của người lạ — đánh giá rejection (testUNKNOWN.ipynb)
└── tool.py        # Script tiện ích: đổi tên ảnh trong People/<tên>/ thành 1.jpg, 2.jpg, ...
```

> Lưu ý: `Dataset/People`, `Dataset/Test`, `Dataset/UNKNOWN-TEST` được liệt kê trong `.gitignore` — đây là dữ liệu cá nhân không commit, phải tự tải từ Drive.

### 1.2. Pretrained Model
Tải weights tại:
https://drive.google.com/file/d/1qcMrQx21Ef7nY27mOHctRxpk_Jaus99t/view?usp=sharing

```bash
# 1. Lưu vào: Model/model_weights.zip
# 2. Giải nén
unzip Model/model_weights.zip -d Model/

# 3. Chuyển từng file weight rời rạc thành 1 file H5 duy nhất
cd Model
python weight_to_h5.py
# => Sinh ra file Model/FaceNet_Keras.h5 dùng cho inference
```

### 1.3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
Thư viện chính: `keras==2.8.0`, `mtcnn==0.1.1`, `opencv-python==4.5.5.64`, `scikit-learn==1.0.2`, `scipy==1.8.0`, `imutils`, `joblib`, `matplotlib`, `seaborn`, `numpy==1.22.3`.

---

## 2. Lý thuyết nhận diện khuôn mặt

Một hệ thống nhận diện khuôn mặt cổ điển gồm 3 giai đoạn nối tiếp nhau:

```
   Ảnh đầu vào
        │
        ▼
┌──────────────────┐
│  Face Detection  │  ◄── MTCNN: xác định bounding box khuôn mặt
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ Face Embedding   │  ◄── FaceNet (Inception-ResNet V1):
│  (Trích đặc      │      ảnh khuôn mặt → vector 128 chiều
│   trưng)         │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  Classification  │  ◄── So sánh vector với Database
│   (Identify)     │      (Euclidean Distance hoặc SVM)
└──────────────────┘
        │
        ▼
   Tên người / UNKNOWN
```

### 2.1. Face Detection — MTCNN (Multi-task Cascaded Convolutional Networks)

**MTCNN** (Zhang et al., 2016) là một kiến trúc *cascade* gồm 3 mạng CNN chạy nối tiếp, từ nhẹ → nặng. Ý tưởng cốt lõi: lọc dần các vùng *không phải mặt* để chỉ những vùng "đáng nghi" mới phải qua mạng phức tạp hơn.

**Bước chuẩn bị — Image Pyramid:**
- Ảnh đầu vào được resize ra nhiều tỉ lệ (vd: 1.0, 0.7, 0.5, ...) tạo thành 1 "kim tự tháp" ảnh.
- Mục đích: phát hiện được mặt ở mọi kích thước mà không cần sliding window đa kích thước.

**Stage 1 — P-Net (Proposal Network):**
- CNN rất nhỏ (≈ vài layer Conv) chạy fully-convolutional trên *từng tầng* của pyramid.
- Output: với mỗi vị trí trả về (face/non-face score, bounding box regression).
- Sinh ra rất nhiều candidate box (hàng ngàn).
- **NMS** (Non-Max Suppression) lọc bớt các box trùng nhau.

**Stage 2 — R-Net (Refine Network):**
- Lớn hơn P-Net. Lấy mỗi candidate từ P-Net, resize về `24×24`, đẩy qua CNN.
- Loại bỏ phần lớn false positive còn sót.
- Refine bounding box (regression) chính xác hơn.
- Lại NMS.

**Stage 3 — O-Net (Output Network):**
- Mạng lớn nhất. Input `48×48`.
- Trả về 3 output cùng lúc (đây là chữ "multi-task" trong tên):
  1. Face / non-face classification
  2. Bounding box regression
  3. **5 facial landmarks** (2 mắt, mũi, 2 khoé miệng)
- NMS lần cuối → bounding box hoàn chỉnh.

**Vì sao chọn MTCNN cho dự án này?**
- **Chính xác hơn Haar Cascade** truyền thống của OpenCV (vốn nhạy với góc nghiêng / ánh sáng).
- **Nhanh đủ để chạy realtime** trên CPU nhờ cascade — phần lớn vùng ảnh bị loại ngay ở P-Net rất nhẹ. Trong code, để tăng tốc thêm, dự án còn resize ảnh xuống **1/4** (`scale=4` trong `Detect_Face`) trước khi đưa vào MTCNN, rồi nhân toạ độ trở lại — đánh đổi 1 chút độ chính xác lấy FPS webcam.
- **Đầu ra là bounding box thuần** → vừa khít với input mà FaceNet cần (`crop` → resize 160×160). Dự án chỉ dùng box, **không dùng 5 landmark** (do đó cũng bỏ qua bước face alignment xoay-thẳng đầu — chấp nhận hi sinh thêm 1 ít accuracy đổi lấy đơn giản).

---

### 2.2. Face Embedding — FaceNet (Inception-ResNet V1)

**FaceNet** (Schroff et al., Google 2015) **không phải** là một classifier theo nghĩa truyền thống. Nó là một **embedding network**: chỉ làm 1 việc duy nhất là biến ảnh khuôn mặt thành 1 vector số thực sao cho **khoảng cách trong không gian vector phản ánh đúng "độ giống nhau" giữa các khuôn mặt**.

#### Triplet Loss — linh hồn của FaceNet
Khi huấn luyện, mỗi sample là 1 *triplet* gồm:
- **Anchor (a)** — 1 ảnh của người X
- **Positive (p)** — 1 ảnh **khác** cũng của người X
- **Negative (n)** — 1 ảnh của người Y ≠ X

Loss function ép vector của anchor gần positive hơn negative ít nhất 1 lượng margin α:

```
L = max( ||f(a) - f(p)||²  -  ||f(a) - f(n)||²  +  α,  0 )
```

Hệ quả sau khi train:
- Cùng 1 người → các vector co cụm vào 1 vùng nhỏ.
- 2 người khác nhau → các vùng cách xa nhau ≥ α.
- Mô hình **chưa bao giờ học các định danh cụ thể** — nó chỉ học "khoảng cách". Đây là lý do quan trọng: ta có thể **thêm người mới mà KHÔNG cần train lại model**, chỉ cần trích vector và lưu vào Database.

> Đây chính là khác biệt căn bản với cách "train 1 CNN classifier với N class = N người": cách kia mỗi lần thêm người là phải retrain. FaceNet thì không.

#### Backbone — Inception-ResNet V1
File `Model/inception_resnet_v1.py` định nghĩa toàn bộ kiến trúc. Tổng quan:

```
Input (160×160×3)
   │
   ▼  Stem: chuỗi Conv2D + BN + ReLU
Conv 32 (s=2) → Conv 32 → Conv 64 → MaxPool
   → Conv 80 → Conv 192 → Conv 256 (s=2)
   │
   ▼
[ Inception-ResNet-A (Block35) ] × 5    scale=0.17
   │
   ▼
Reduction-A (Mixed_6a)
   │
   ▼
[ Inception-ResNet-B (Block17) ] × 10   scale=0.10
   │
   ▼
Reduction-B (Mixed_7a)
   │
   ▼
[ Inception-ResNet-C (Block8) ]  × 5    scale=0.20
   │   + 1 Block8 cuối không activation
   ▼
GlobalAveragePooling2D
   │
   ▼
Dropout (keep=0.8)
   │
   ▼
Dense(128, no bias)  ← "Bottleneck" — chính là vector đặc trưng
   │
   ▼
BatchNorm  →  Output (128-D)
```

Vì sao kết hợp **Inception + ResNet**?
- **Inception block** (nhánh `1×1`, `3×3`, `5×5`/`1×7+7×1` song song rồi concat) giúp model nhìn được đặc trưng ở **nhiều scale cùng lúc** trong cùng 1 layer — cần thiết cho khuôn mặt có nhiều chi tiết kích cỡ khác nhau (mắt, mũi vs. cấu trúc tổng thể khuôn mặt).
- **Residual connection** (`x + branch(x)`, có nhân `scale` để ổn định khi cộng) giúp gradient truyền tốt qua mạng sâu, tránh vanishing gradient — cho phép train mạng rất sâu hội tụ ổn định.
- 3 kiểu Block (35/17/8) tương ứng kích thước feature map giảm dần (35×35 → 17×17 → 8×8) khi xuống sâu, mỗi block tinh chỉnh đặc trưng ở mức độ trừu tượng khác nhau.

#### L2 Normalization — bước cuối cùng
Sau khi model trả về vector 128-D, code gọi `L2_Normalize` để chia vector cho độ dài của nó:

```
v_norm = v / ||v||₂
```

Sau bước này, mọi embedding đều **nằm trên mặt cầu đơn vị** (`||v_norm|| = 1`). Có 2 hệ quả thú vị:
1. **Euclidean distance² và cosine similarity tương đương** (chỉ khác hệ số):
   ```
   ||a - b||² = ||a||² + ||b||² - 2·a·b = 2 - 2·cos(a, b)
   ```
   ⇒ Khoảng cách Euclidean nằm trong khoảng `[0, 2]` thay vì vô hạn — dễ chọn threshold.
2. **Bất biến với độ sáng / scale** — chỉ còn quan tâm hướng vector, không quan tâm độ lớn.

#### Vì sao chọn FaceNet cho dự án?
- **Pretrained available** — model trên `Inception-ResNet V1` đã được train sẵn trên dataset lớn (MS-Celeb-1M, VGGFace2), chỉ cần download weight về và inference.
- **128-D nhỏ gọn** — toàn bộ Database 20 người × 5 vector chỉ là một file JSON vài chục KB.
- **Không cần retrain khi thêm người** — đặc tính của embedding learning (xem Triplet Loss ở trên).
- **State-of-the-art** thời điểm 2015–2018 trên LFW (99.6%+) — đủ dùng cho ứng dụng demo / hệ thống nhỏ.

---

### 2.3. Classification — Phân loại danh tính

Sau khi có vector 128-D đã L2-normalize, vấn đề trở thành: "vector này gần với *cluster* của ai nhất trong Database, và liệu có đủ gần để khẳng định không?"

Dự án cài đặt **2 cách giải song song**:

#### Cách 1 — Euclidean Distance (1-NN brute force) — `Face_Identify`

```python
# Pseudocode — Inference/Facenet.py:28
for person, vectors in database.items():
    d[person] = min( euclidean(v_query, v) for v in vectors )

name = argmin(d)
if d[name] > 1.0:
    name = "UNKNOWN"
```

- **Bản chất**: 1-Nearest-Neighbor — chọn người có vector mẫu *gần nhất* với query.
- Lấy `min` qua tất cả 5 vector của 1 người (không phải mean) → robust với 1-2 ảnh enroll "xấu".
- Threshold **1.0** chọn theo kinh nghiệm (vì vector L2-normalized nên distance ∈ [0, 2]).

**Ưu điểm**: không cần train gì cả, thêm người là dùng được ngay.
**Nhược điểm**: khi số người tăng, ranh giới giữa các cluster có thể chồng lấn, và 1-NN dễ "lệch" bởi 1 vector mẫu nhiễu duy nhất.

#### Cách 2 — Linear SVM + Distance threshold — `Get_People_Identity_SVM`

```python
# Pseudocode — Inference/Facenet.py:73
name = svm.predict(v_query)[0]                  # SVM gán nhãn
dist = min( euclidean(v_query, v)               # distance đến cluster của nhãn đó
            for v in database[name] )
if dist > 0.7:
    name = "UNKNOWN"
```

**Vì sao dùng SVM tuyến tính?**
- Sau khi đã embed bằng FaceNet, các cluster trong không gian 128-D thường đã **gần như linearly separable** (Triplet Loss ép chúng cách xa nhau). Một SVM tuyến tính nhỏ là đủ → khỏi dùng kernel phức tạp.
- SVM tối đa hoá **margin** giữa các lớp ⇒ ranh giới quyết định ổn định hơn 1-NN khi có vector nhiễu.
- Train rất nhanh (vài giây cho 100 vector), file `svm.sav` chỉ vài KB.
- Khi số người tăng dần, SVM mở rộng tốt hơn 1-NN (1-NN tốn O(N) so sánh cho mỗi query, SVM tuyến tính chỉ là 1 phép nhân ma trận).

**Vì sao vẫn cần threshold distance dù đã có SVM?**
SVM **luôn luôn trả về 1 nhãn**, kể cả khi đưa vào ảnh của người lạ. Không có khái niệm "tôi không biết" nội tại. Vậy nếu 1 người lạ xuất hiện, SVM sẽ gán đại 1 nhãn nào đó (gần nhất theo decision boundary) → false positive.

Giải pháp: sau khi SVM gán nhãn, ta vẫn đo **khoảng cách Euclidean từ query đến cluster của nhãn đó**. Nếu khoảng cách quá xa (`> 0.7`) ⇒ chắc chắn không phải nhãn đó dù SVM "khăng khăng" → ghi đè thành `UNKNOWN`.

→ SVM giải bài toán **"ai trong số những người đã biết"**, distance threshold giải bài toán **"hay đây là người lạ?"**. Hai tầng kết hợp = **closed-set classification + open-set rejection**.

**Threshold 0.7 < 1.0**: chặt hơn cách 1 — vì đã có SVM phân lớp nên ta có thể siết threshold xuống để giảm false positive (chấp nhận tăng nhẹ false reject).

### 2.4. Tiền xử lý ảnh trước khi đưa vào FaceNet
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — cân bằng sáng cục bộ trên kênh `Y` (luminance) trong không gian YUV. Giúp ổn định kết quả khi ảnh thiếu/thừa sáng.
2. **Resize** về `160 × 160`.
3. **Per-image standardization**: chuẩn hoá mean = 0, std = 1 từng ảnh (dùng `std_adj = max(std, 1/sqrt(N))` để tránh chia 0).
4. **Predict** qua model → vector 128D.
5. **L2 Normalize** vector trước khi so sánh.

---

## 3. Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|------------|-----------|---------|
| Face detection | `mtcnn` (Keras-based MTCNN) | Phát hiện bounding box khuôn mặt |
| Face embedding | `keras` + Inception-ResNet V1 weights | Trích vector đặc trưng 128D |
| Image processing | `opencv-python` | Đọc ảnh, resize, đổi không gian màu, CLAHE |
| Numeric / distance | `numpy`, `scipy.spatial.distance` | Tính Euclidean distance |
| Classifier | `scikit-learn` `SVC(kernel="linear")` | Phân loại danh tính |
| Persistence | `joblib`, `json` | Lưu SVM model (`.sav`) và Database vector (`.json`) |
| Visualization | `matplotlib`, `seaborn`, `sklearn.PCA` | Heatmap confusion matrix, scatter 3D PCA của embedding |
| Helper | `imutils` | Liệt kê đường dẫn ảnh trong dataset |

---

## 4. Cấu trúc thư mục

```
Face-Recognition/
├── Model/
│   ├── inception_resnet_v1.py     # Định nghĩa kiến trúc backbone
│   ├── weight_to_h5.py            # Build kiến trúc + nạp weight + lưu .h5
│   └── FaceNet_Keras.h5           # File model dùng cho inference
├── Inference/
│   ├── FaceDetection.py           # Wrapper MTCNN: detect + crop + draw box
│   ├── FaceRecognition.py         # Tiền xử lý ảnh + sinh embedding
│   └── Facenet.py                 # Pipeline chính: detect → embed → identify
├── Classifier/
│   └── SVM.py                     # Train / load / visualize SVM
├── Database/
│   ├── FeatureVector.py           # Đọc Dataset/People, sinh embedding, lưu JSON
│   └── Database.json              # {tên_người: [vector_128D, ...], ...}
├── SVM_Model/
│   ├── svm.sav                    # SVM đã huấn luyện (joblib)
│   ├── train_confusion.png        # Confusion matrix tập train
│   └── test_confusion.png         # Confusion matrix tập test
├── Dataset/
│   ├── People/                    # Enroll dataset
│   ├── Test/                      # Test dataset (đã biết danh tính)
│   ├── UNKNOWN/                   # Người lạ (kiểm tra rejection)
│   └── tool.py                    # Đổi tên file ảnh
├── TestWebcam.py                  # Demo webcam — 1 khuôn mặt
├── TestMultiPerson.py             # Demo webcam — nhiều khuôn mặt cùng lúc
├── API.ipynb                      # Notebook chạy thử pipeline đầu - cuối
├── testBackbone.ipynb             # Kiểm tra summary của FaceNet model
├── testSVM.ipynb                  # Đánh giá SVM trên Dataset/Test
├── testUNKNOWN.ipynb              # Đánh giá rejection trên Dataset/UNKNOWN
├── Visualize.ipynb                # PCA 3D embedding + so sánh equalizeHist vs CLAHE
└── requirements.txt
```

---

## 5. Luồng hoạt động chi tiết

### 5.1. Giai đoạn ENROLL — xây Database từ ảnh người
Chạy 1 lần khi muốn thêm/cập nhật người vào hệ thống.

```python
# API.ipynb — cell đầu tiên
from Database.FeatureVector import FeatureVector

sfv = FeatureVector("Dataset/People")
sfv.Check_Database()        # Tạo file Database.json rỗng nếu chưa có
sfv.Get_People_Feature()    # Trích vector cho từng ảnh, lưu vào DB
```

Bên trong `FeatureVector.Get_People_Feature()` (`Database/FeatureVector.py:67`):
1. `Load_People_Img()` — duyệt tất cả ảnh trong `Dataset/People/<tên>/*.jpg`. Tên thư mục cha = nhãn (`ip.split(sep)[-2]`).
2. Với mỗi ảnh:
   - `FaceDetection.Detect_Face(img, resize=True, scale=4)` → bounding box.
   - `FaceDetection.Crop_Face(img, rec)[0]` → lấy khuôn mặt **đầu tiên** (giả định mỗi ảnh enroll chỉ có 1 mặt).
   - `FaceRecognition.Get_Face_Embedding(face)` → vector 128D.
   - `Save_Feature_To_Database(name, vector)` → append vào `Database.json`.

Kết quả: `Database.json` có dạng
```json
{
  "Phuoc":  [[0.037, -0.095, ...], [...], ...],   // 5 vector 128D
  "Hoang":  [[...], ...],
  ...
}
```
Hiện tại Database có 20 người × 5 vector/người (xem `Database/Database.json`).

### 5.2. Giai đoạn TRAIN SVM
Sau khi Database đã đủ vector, huấn luyện SVM để tăng độ chính xác phân lớp.

```python
# API.ipynb — cell 7
from Classifier.SVM import SVM
svm = SVM()
svm.train()
```

Bên trong `SVM.train()` (`Classifier/SVM.py:47`):
1. `split_data(test_size=0.2)` — chia stratified theo từng người (đảm bảo mỗi người đều có train+test).
2. Tạo `SVC(kernel="linear", probability=True)` → `fit(X_train, Y_train)`.
3. Dự đoán trên train/test → tính accuracy → vẽ và lưu 2 confusion matrix vào `SVM_Model/`.
4. Dump model bằng `joblib` ra `SVM_Model/svm.sav`.

### 5.3. Giai đoạn INFERENCE — nhận diện
Có 2 entry-point: webcam realtime (`TestWebcam.py`, `TestMultiPerson.py`) và notebook (`API.ipynb`). Cả hai đều gọi cùng 1 hàm trong `Facenet`:

```python
fn = Facenet()
people = fn.Get_People_Identity_SVM(img)
#   => [(name, distance, embedding, [x1,y1,x2,y2]), ...]
```

`Facenet.__init__()` (`Inference/Facenet.py:10`) khởi tạo 3 thành phần:
- `FaceDetection()` — load MTCNN.
- `FaceRecognition()` — load `Model/FaceNet_Keras.h5` + tạo CLAHE.
- `SVM()` — chỉ đọc `Database.json`, model SVM được lazy-load mỗi lần predict.

#### Pipeline 1 frame ảnh

```
img (BGR) ─┐
           ▼
┌───────────────────────────────────────────────────┐
│ FaceDetection.Detect_Face(img, resize=True, s=4)  │  Inference/FaceDetection.py:10
│  • copy ảnh → resize xuống 1/4 cho nhanh          │
│  • BGR → RGB (MTCNN yêu cầu RGB)                  │
│  • detector.detect_faces(rgb) → list face dict    │
│  • Trả về list [x1,y1,x2,y2] đã nhân lại scale=4  │
└───────────────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────┐
│ FaceDetection.Crop_Face(img, rec)                 │  FaceDetection.py:49
│  • img[y1:y2, x1:x2] cho mỗi box                  │
└───────────────────────────────────────────────────┘
           │ (lặp qua từng face)
           ▼
┌───────────────────────────────────────────────────┐
│ FaceRecognition.Get_Face_Embedding(face)          │  FaceRecognition.py:64
│  1. Adaptive_Histogram_Equalization (CLAHE on Y)  │
│  2. Preprocessing_IMG:                            │
│       resize 160×160 → mean/std standardize       │
│  3. expand_dims → model.predict → vector 128D     │
│  4. L2_Normalize                                  │
└───────────────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────┐
│ Classifier — Cách 2 (mặc định):                   │  Facenet.py:73
│  svm = SVM.load_model()                           │
│  name = svm.predict(embedding)[0]                 │
│  distance = min(Euclidean(embd, v)                │
│                  for v in database[name])         │
│  if distance > 0.7: name = "UNKNOWN"              │
│                                                    │
│ Classifier — Cách 1 (Get_People_Identity):        │  Facenet.py:28
│  Với mỗi person trong DB:                         │
│     d[person] = min(Euclidean(embd, v) for v...)  │
│  name = argmin(d); if d[name] > 1.0 → UNKNOWN     │
└───────────────────────────────────────────────────┘
           │
           ▼
   [(name, distance, embedding, box), ...]
```

#### Vẽ kết quả lên frame (TestWebcam/TestMultiPerson)
```python
for identity, distance, _, box in people_list:
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 1)
    cv2.putText(img, identity, (box[0], box[1]), ..., (0,255,0), 2)
```
Nhấn `q` để thoát.

---

## 6. Cách chạy

### 6.1. Realtime webcam (1 khuôn mặt — lấy mặt đầu tiên)
```bash
python TestWebcam.py
```

### 6.2. Realtime webcam (nhiều khuôn mặt)
```bash
python TestMultiPerson.py
```

### 6.3. Notebook
- `API.ipynb` — chạy đầy đủ chu trình: enroll DB → train SVM → predict 1 ảnh.
- `testSVM.ipynb` — chạy đánh giá toàn bộ `Dataset/Test`, in accuracy + confusion matrix.
- `testUNKNOWN.ipynb` — chạy đánh giá rejection trên `Dataset/UNKNOWN-TEST`, kì vọng đầu ra "UNKNOWN".
- `testBackbone.ipynb` — in `model.summary()` của FaceNet để kiểm tra input/output shape.
- `Visualize.ipynb` — giảm vector 128D → 3D bằng **PCA**, vẽ scatter 3D từng người; so sánh `equalizeHist` vs `CLAHE` trực quan.

---

## 7. Các tham số / ngưỡng đáng chú ý

| Tham số | Vị trí | Giá trị | Vai trò |
|---------|--------|---------|---------|
| `scale` (detect) | `FaceDetection.Detect_Face` | `4` | Resize ảnh xuống 1/4 trước khi đưa MTCNN ⇒ nhanh hơn |
| `clipLimit` | `FaceRecognition.Create_Clahe` | `1.5` | Mức giới hạn tương phản của CLAHE |
| `tileGridSize` | CLAHE | `(8, 8)` | Kích thước ô lưới cân bằng cục bộ |
| Input shape | FaceNet | `160 × 160 × 3` | Yêu cầu của Inception-ResNet V1 |
| Embedding size | FaceNet | `128` | Kích thước vector đặc trưng |
| `min_dist > 1.0` | `Facenet.Face_Identify` | `1.0` | Ngưỡng UNKNOWN khi dùng Euclidean trực tiếp |
| `distance > 0.7` | `Facenet.Get_People_Identity_SVM` | `0.7` | Ngưỡng UNKNOWN khi đã dùng SVM (chặt hơn) |
| `kernel` (SVM) | `Classifier/SVM.py` | `"linear"` | SVM tuyến tính + `probability=True` |
| `test_size` | `SVM.split_data` | `0.2` | Tỉ lệ split train/test |

---

## 8. Tóm tắt một câu

> Detect mặt bằng MTCNN → nhúng ảnh khuôn mặt thành vector 128D bằng FaceNet (Inception-ResNet V1) đã pretrained → so vector với Database 20 người (Euclidean distance) hoặc đẩy qua SVM tuyến tính, kèm ngưỡng distance để từ chối người lạ (UNKNOWN).
