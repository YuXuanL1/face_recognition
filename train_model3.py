#  1. 資料增強與切分
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

data_dir = r'C:\Users\6yx\Downloads\Face_Recognition_w_DeepFace-master\Face_Recognition_w_DeepFace-master\Data'
classes = os.listdir(data_dir)

# 定義數據增強策略
datagen = ImageDataGenerator(
    rotation_range=20,  # 隨機旋轉
    width_shift_range=0.2,  # 隨機水平平移
    height_shift_range=0.2,  # 隨機垂直平移
    shear_range=0.2,  # 隨機錯切變換
    zoom_range=0.2,  # 隨機縮放
    horizontal_flip=True,  # 隨機水平翻轉
    fill_mode='nearest'  # 填充空缺
)

# 加載圖像資料和標籤
def load_images_and_labels(data_dir):
    images = []
    labels = []
    for label, person in enumerate(classes):
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):  # 確認是目錄
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:  # 確保圖像讀取成功
                    img = cv2.resize(img, (224, 224))  # 調整圖片大小
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)


# 加載數據
images, labels = load_images_and_labels(data_dir)

# 切分資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 對訓練集進行數據增強
train_generator = datagen.flow(X_train, y_train, batch_size=32)

# # 
# # 獲取訓練集的原始大小
# original_size = len(X_train)
# print(f"原始訓練集數據數量: {original_size}")

# # 設定要生成的批次數量
# num_batches = 32  # 獲取 n 個批次的增強數據

# # 計算增強數據的總數
# augmented_images = []
# augmented_labels = []

# for _ in range(num_batches):
#     images, labels = next(train_generator)  # 獲取一批增強數據
#     augmented_images.append(images)
#     augmented_labels.append(labels)

# # 將所有增強數據合併
# augmented_images = np.vstack(augmented_images)
# augmented_labels = np.hstack(augmented_labels)

# # 打印增強後的數據數量
# augmented_size = augmented_images.shape[0]
# print(f"增強後的訓練集數據數量: {augmented_size}")
# # 


# 2. 遷移學習
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# 加載預訓練的 ResNet50 模型，並保留預訓練權重
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 凍結預訓練模型的層
for layer in base_model.layers:
    layer.trainable = False

# 添加自定義的分類層
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

# 定義遷移學習的模型
model = Model(inputs=base_model.input, outputs=predictions)

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(train_generator, validation_data=(X_test, y_test), epochs=50)


# 3.模型評估S
score = model.evaluate(X_test, y_test, verbose=0)
print(f"測試集準確率: {score[1] * 100:.2f}%")

# 保存訓練好的模型
model.save('face_recognition_model3.h5')
print("模型已保存到 'face_recognition_model3.h5'")