import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# 定義圖像預處理函數
def preprocess_image(image_path):
    # 加載 Haar 級聯分類器(偵測人臉)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 讀取圖像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 檢測人臉
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # 如果檢測到人臉，則裁剪並返回處理過的圖像
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = img[y:y+h, x:x+w]

        # 調整圖像大小為 MobileNetV3 所需的 224x224
        face_img = cv2.resize(face_img, (224, 224))
        
        # 將BGR轉換為RGB格式
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # 將圖像轉換為數組格式
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)  # 添加 batch 維度

        return face_img
    else:
        print(f"No faces detected in {image_path}")
        return None

# 讀取資料夾並進行預處理
def load_dataset(data_dir):
    X_train = []
    y_train = []
    
    # 遍歷資料夾中的每個人的資料夾
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                
                # 預處理每張圖片
                processed_image = preprocess_image(img_path)
                if processed_image is not None:
                    X_train.append(processed_image)
                    y_train.append(person_name)
    
    # 將 X_train 合併為一個 numpy 陣列
    X_train = np.vstack(X_train)
    
    # 將標籤轉為數值格式
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    
    # 將標籤轉為 one-hot 編碼格式
    y_train = to_categorical(y_train)
    
    return X_train, y_train, label_encoder

# 保存處理過後的數據
def save_dataset(X_train, y_train, label_encoder):
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    
    with open('label_encoder.npy', 'wb') as f:
        np.save(f, label_encoder.classes_)

# 主程序：加載數據並保存
if __name__ == "__main__":
    data_dir = "Face_Recognition_w_DeepFace-master\Data"  # 人臉圖像資料夾的路徑
    X_train, y_train, label_encoder = load_dataset(data_dir)
    
    # 保存預處理後的數據
    save_dataset(X_train, y_train, label_encoder)
    print("Dataset processing completed and saved.")



# 加載之前保存的數據
def load_saved_dataset():
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    label_encoder = np.load('label_encoder.npy', allow_pickle=True)
    return X_train, y_train, label_encoder

# 構建遷移學習模型 (使用 MobileNetV3)
def create_model(num_classes, version="large"):
    # 根據選擇的版本加載 MobileNetV3 模型
    if version == "large":
        base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 添加新的分類層
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # 定義新模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 鎖住預訓練模型的所有層，只訓練新添加的層
    for layer in base_model.layers:
        layer.trainable = False

    # 編譯模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# 主程序：加載數據並訓練模型
if __name__ == "__main__":
    # 加載保存的數據
    X_train, y_train, label_encoder = load_saved_dataset()
    
    # 確定類別數
    num_classes = len(label_encoder)
    
    # 構建模型，這裡選擇 large 版 MobileNetV3
    model = create_model(num_classes, version="large")
    
    # 訓練模型
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    
    # 保存訓練好的模型
    model.save('face_recognition_mobilenetv3.keras')
    print("Model training completed and saved.")
