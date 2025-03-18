from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from fastapi import FastAPI
import threading
import time
from collections import Counter

# List of available backends, models, and distance metrics
# backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
# models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
# metrics = ["cosine", "euclidean", "euclidean_l2"]

recognition_results = {"name": "未知"}  # 存儲最終結果

def get_name_from_path(path):

    # 獲取資料夾路徑
    dir_path = os.path.dirname(path)
    # 分割路徑，取最後一個部分作為名稱
    return os.path.basename(dir_path)

def realtime_face_recognition():
    global recognition_results
    vid = cv2.VideoCapture(0)
    names_detected = []
    start_time = time.time()

    while True:
        ret, frame = vid.read()

        try:
            people = DeepFace.find(img_path=frame, db_path="Face_Recognition_w_DeepFace-master\Data", model_name="SFace", distance_metric="cosine", enforce_detection=False)
            
            if isinstance(people, list) and len(people) > 0 and isinstance(people[0], pd.DataFrame):
                for person in people:
                    if not person.empty:
                        for _, row in person.iterrows():
                            x = int(row['source_x'])
                            y = int(row['source_y'])
                            w = int(row['source_w'])
                            h = int(row['source_h'])

                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                            name = "未知"
                            if 'identity' in row:
                                identity = row['identity']
                                if isinstance(identity, str):
                                    name = get_name_from_path(identity)
                                elif isinstance(identity, list) and len(identity) > 0:
                                    name = get_name_from_path(identity[0])
                            
                            names_detected.append(name)
                            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"發生錯誤: {e}")

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 960, 720)
        cv2.imshow('frame', frame)

        # 檢查是否已經過了5秒，並回傳5秒內最常出現的人名
        if time.time() - start_time >= 10:
            if names_detected:
                most_common_name = Counter(names_detected).most_common(1)[0][0]
                recognition_results["name"] = most_common_name
                print(f"偵測結果: {most_common_name}")
            break  # 偵測5秒後退出循環

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

# 加上transfer learning
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.optimizers import Adam

# # 加載預訓練的模型，如 ArcFace
# base_model = DeepFace.build_model("SFace")

# # 凍結預訓練模型的前幾層，只訓練最後幾層
# for layer in base_model.layers[:-4]:
#     layer.trainable = False

# # 添加新的全連接層
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)  # 新的全連接層
# predictions = Dense(num_classes, activation='softmax')(x)  # 根據新數據集的類別數量調整輸出層

# # 定義新模型
# model = Model(inputs=base_model.input, outputs=predictions)

# # 編譯模型
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# # 用新的數據進行微調訓練
# model.fit(new_data, new_labels, epochs=10, batch_size=32)


# Perform real-time face recognition using the webcam
realtime_face_recognition()

# # 串API
# app = FastAPI()

# @app.get("/recognize_and_get_result")
# def recognize_and_get_result():
#     # 啟動即時人臉辨識
#     realtime_face_recognition()
#     # 返回結果
#     return recognition_results