import cv2
import time
import numpy as np
from keras.models import load_model
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# 載入訓練好的MobileNetV2模型
model = load_model(r"Face_Recognition_w_DeepFace-master\face_recognition_mobilenetv3.keras")

# 載入Label Encoder來轉換預測標籤
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(r"Face_Recognition_w_DeepFace-master\label_encoder.npy")  # 假設有個classes.npy保存了類別名稱

# OpenCV的人臉檢測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    """對輸入的圖像進行預處理（調整大小、轉換格式等）"""
    face_img = cv2.resize(image, (224, 224))  # MobileNetV2需要224x224的輸入大小
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype('float32') / 255.0  # 正規化到[0, 1]
    face_img = np.expand_dims(face_img, axis=0)  # 添加批次維度
    return face_img

def realtime_face_recognition():
    vid = cv2.VideoCapture(0)  # 使用攝像頭
    names_detected = []
    start_time = time.time()

    while True:
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 偵測人臉
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_input = preprocess_image(face)

            # 使用MobileNetV2模型進行預測
            preds = model.predict(face_input)
            name_idx = np.argmax(preds)  # 預測得分最高的類別
            name = label_encoder.inverse_transform([name_idx])[0]  # 將預測的標籤轉換回人名

            # 在畫面上畫出人臉框和名字
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 保存檢測到的名字
            names_detected.append(name)

        # 顯示即時畫面
        cv2.imshow('frame', frame)

        # 偵測5秒內出現最多次的名字並回傳
        if time.time() - start_time >= 10:
            if names_detected:
                most_common_name = Counter(names_detected).most_common(1)[0][0]
                print(f"偵測結果: {most_common_name}")
            break  # 5秒後結束循環

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

# 啟動即時人臉識別
realtime_face_recognition()
