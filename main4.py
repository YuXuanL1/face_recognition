import cv2
import numpy as np
import torch
import threading
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from torchvision import transforms
from facenet_pytorch import MTCNN
from pathlib import Path
import os
from datetime import datetime, timezone
from collections import Counter
import httpx
import shutil
import time

# 初始化 FastAPI 應用
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 根據需求設定允許的來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域變數
recognition_results = {"name": "未知"}  # 儲存最終的辨識結果

# 設定裝置（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 請確保 model.pt 路徑正確
model_path = '/app/model/model.pt'
model = torch.jit.load(model_path, map_location=device)
model.eval()

# 初始化 MTCNN 用於人臉偵測
mtcnn = MTCNN(keep_all=True, device=device)

# 特徵資料庫 (字典)：{ 'phone_number': tensor_features }
feature_db = {}

def load_feature_db():
    global feature_db
    data_dir = Path('/app/faceModel/Data')
    if not data_dir.exists():
        print(f"資料目錄 {data_dir} 不存在。")
        return
    for user_dir in data_dir.iterdir():
        if user_dir.is_dir():
            phone_number = user_dir.name
            feature_files = list(user_dir.glob('*.npy'))
            if feature_files:
                # 假設每位用戶只有一個特徵檔案
                feature_file = feature_files[0]
                features = np.load(str(feature_file))
                feature_db[phone_number] = torch.tensor(features).to(device)
            else:
                print(f"用戶 {phone_number} 的特徵檔案不存在。")

load_feature_db()

def preprocess(face):
    # 將圖片轉為 RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # PIL 圖片轉換
    face = transforms.ToPILImage()(face)
    # 定義轉換
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(face).unsqueeze(0).to(device)

def extract_features(face):
    with torch.no_grad():
        img_tensor = preprocess(face)
        features = model(img_tensor)
    return features

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b).item()

def realtime_face_recognition():
    global recognition_results
    vid = cv2.VideoCapture('/dev/video0')
    if not vid.isOpened():
        print("無法打開攝影機")
        return
    else:
        print("攝影機已打開")

    names_detected = []
    start_time = time.time()

    while True:
        ret, frame = vid.read()
        if not ret:
            print("無法讀取影像幀")
            break

        try:
            # 人臉偵測
            boxes, _ = mtcnn.detect(frame)
            # 確保 boxes 不為 None 且有資料
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face = frame[y1:y2, x1:x2]
                    # 特徵提取
                    features = extract_features(face)
                    # 與資料庫比對
                    best_match = None
                    highest_score = 0.0
                    for phone_number, db_features in feature_db.items():
                        score = cosine_similarity(features, db_features)
                        if score > highest_score:
                            highest_score = score
                            best_match = phone_number
                    # 設定閾值，例如 0.5
                    if highest_score > 0.5 and best_match is not None:
                        names_detected.append(best_match)
                        # 若在無圖形介面的環境下，不需顯示圖像窗口
                        # 但若要在本地測試可視化，請確保有 GUI 環境才使用下列代碼
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # cv2.putText(frame, best_match, (x1, y1 - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        # 未達閾值或未知對象
                        pass
                # 假如想要在有 GUI 的環境查看結果，可取消註解：
                # cv2.imshow('frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                # 未偵測到人臉，不做任何處理
                pass

        except Exception as e:
            print(f"發生錯誤: {e}")

        # 每 10 秒結束循環，回傳最常出現的名稱
        if time.time() - start_time >= 10:
            if names_detected:
                most_common_name = Counter(names_detected).most_common(1)[0][0]
                recognition_results["name"] = most_common_name
                print(f"偵測結果: {most_common_name}")
            else:
                recognition_results["name"] = "未知"
            break  # 跳出循環

    vid.release()
    # 不要在無圖形介面下使用下列函式
    # cv2.destroyAllWindows()

async def send_user_info_to_nodejs(name: str, phone: str):
    url = "http://140.119.19.85:3000/api/1.0/user/faceSignup"
    payload = {
        "name": name,
        "phone": phone
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
            print(f"成功將用戶資訊發送到 NodeJS 後端：{response.status_code}")
    except httpx.HTTPStatusError as http_err:
        print(f"HTTP 錯誤發生: {http_err.response.status_code} - {http_err.response.text}")
    except Exception as err:
        print(f"其他錯誤發生: {err}")

async def send_face_signin_to_nodejs(recognizedPhone: str):
    url = "http://140.119.19.85:3000/api/1.0/user/faceSignin"
    payload = {
        "recognizedPhone": recognizedPhone
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
            print(f"成功將辨識結果發送到 NodeJS 後端：{response.status_code}")
            return response.json()
    except httpx.HTTPStatusError as http_err:
        print(f"HTTP 錯誤發生: {http_err.response.status_code} - {http_err.response.text}")
        raise HTTPException(status_code=500, detail="NodeJS 後端服務錯誤。")
    except Exception as err:
        print(f"其他錯誤發生: {err}")
        raise HTTPException(status_code=500, detail="無法與 NodeJS 後端通信。")

def start_recognition_thread():
    recognition_thread = threading.Thread(target=realtime_face_recognition)
    recognition_thread.start()
    return recognition_thread

@app.get("/face/test")
def read_root():
    return {"Hello": "World"}

@app.get("/face/recognize_and_get_result")
async def recognize_and_get_result():
    # 啟動即時人臉辨識執行緒
    thread = start_recognition_thread()
    thread.join()  # 等待辨識執行完成

    recognized_name = recognition_results.get("name", "未知")
    print(f"辨識出的名稱: {recognized_name}")

    if recognized_name == "未知":
        raise HTTPException(status_code=404, detail="未識別到任何用戶。")

    # 將辨識結果發送到 NodeJS 後端進行登入
    user_info = await send_face_signin_to_nodejs(recognized_name)
    print(user_info)

    return user_info

@app.post("/face/register")
async def register_user(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    phone: str = Form(...),
    photo: UploadFile = File(...)
):
    # 驗證電話格式
    if not phone.isdigit():
        raise HTTPException(status_code=400, detail="電話號碼必須為數字。")

    # 定義資料夾路徑
    data_dir = Path('/app/faceModel/Data')
    user_dir = data_dir / phone

    # 檢查是否已註冊
    if user_dir.exists():
        print("此電話號碼已經註冊過。")
        raise HTTPException(status_code=400, detail="此電話號碼已經註冊過。")

    try:
        # 創建用戶資料夾
        user_dir.mkdir(parents=True, exist_ok=False)

        # 讀取上傳的照片
        content = await photo.read()
        file_extension = os.path.splitext(photo.filename)[1].lower()
        if file_extension not in [".jpg", ".jpeg", ".png"]:
            raise HTTPException(status_code=400, detail="不支持的圖片格式。僅支援 JPG、JPEG、PNG。")

        # 解碼圖片
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="無法讀取上傳的圖片。")

        # 人臉偵測
        boxes, _ = mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0])
            face = image[y1:y2, x1:x2]
            # 特徵提取
            features = extract_features(face).cpu().numpy()
            # 保存特徵檔案
            feature_filename = f"{phone}_features.npy"
            np.save(user_dir / feature_filename, features)
            # 更新特徵資料庫
            feature_db[phone] = torch.tensor(features).to(device)
        else:
            raise HTTPException(status_code=400, detail="未偵測到人臉，請上傳清晰的正面人臉照片。")

        # 背景任務發送用戶資訊到 NodeJS 後端
        background_tasks.add_task(send_user_info_to_nodejs, name, phone)

        return {"message": "用戶註冊成功。"}

    except Exception as e:
        # 如果發生錯誤，刪除已創建的目錄
        if user_dir.exists():
            shutil.rmtree(user_dir)
        raise HTTPException(status_code=500, detail=f"註冊失敗：{str(e)}")
