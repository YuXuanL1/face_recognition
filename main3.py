import os
import cv2
import numpy as np
from keras.models import load_model
from collections import Counter
import time
from fastapi import FastAPI
from threading import Thread

# 定義類別名稱
data_dir = r'C:\Users\6yx\Downloads\Face_Recognition_w_DeepFace-master\Face_Recognition_w_DeepFace-master\Data'
classes = os.listdir(data_dir)

# 加載已訓練好的模型
model = load_model('face_recognition_model3.h5')
print("模型已成功加載")

recognition_results = {"name": "未知"}  # 存儲最終結果

# 4.即時辨識
def realtime_face_recognition(model):
    global recognition_results

    video_path = '/app/test.mp4'
    if not os.path.exists(video_path):
        print(f"文件 {video_path} 不存在")
        return
    vid = cv2.VideoCapture('/dev/video0')
    if not vid.isOpened():
        print("can't open camera")
        return
    else:
        print("camera opened successfully")
    names_detected = []
    start_time = time.time()

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # 將捕捉到的影像轉換成模型輸入所需的格式
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)

        # 使用模型進行預測
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])

        # 根據預測結果顯示對應的標籤
        if predicted_class < len(classes):
            label = classes[predicted_class]
        else:
            label = "未註冊過"
            
        names_detected.append(label)  # 儲存識別結果

        # 在影像上繪製標籤
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # cv2.namedWindow('Real-Time Face Recognition', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Real-Time Face Recognition', 960, 720)
        # cv2.imshow("Real-Time Face Recognition", frame)

        # 檢查是否已經過了10秒，並回傳10秒內最常出現的人名
        if time.time() - start_time >= 10:
            if names_detected:
                most_common_name = Counter(names_detected).most_common(1)[0][0]
                recognition_results["name"] = most_common_name
                print(f"偵測結果: {most_common_name}")
            break  # 偵測10秒後退出循環

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

# 使用即時辨識
# realtime_face_recognition_with_model(model)

# 串API
app = FastAPI()
def start_recognition_thread():
    # 啟動獨立執行緒來執行人臉辨識
    recognition_thread = Thread(target=realtime_face_recognition, args=(model,))
    recognition_thread.start()
    return recognition_thread

@app.get("/face/test")
def read_root():
    return {"Hello": "World"}

@app.get("/face/recognize_and_get_result")
def recognize_and_get_result():
    # 啟動即時人臉辨識（在獨立執行緒中執行）
    thread = start_recognition_thread()
    thread.join()  # 等待辨識執行完成
    # 返回結果
    return recognition_results