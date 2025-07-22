# face_recognition
To adapt to the resource limitations in rural areas and enable the use of face recognition technology for fast and accurate customer identification under insufficient computing resources, we have successfully integrated this technology with lightweight hardware devices like Raspberry Pi, leveraging Edge AI and edge computing to achieve efficient and effective operation. This approach ensures real-time processing and enhanced performance even in resource-constrained environments.

## Face Detection:
We used the MTCNN from the facenet_pytorch package for face detection, which returns an array of boxes containing face bounding box coordinates.

## Feature Extraction:
The image is first converted to RGB format using OpenCV (an open-source computer vision and image processing library). Then, torchvision.transforms is used to preprocess the face image (resizing, normalization). Finally, the MobileFaceNet model is employed for feature extraction, returning a face feature vector.

## Face Recognition:
Cosine similarity is calculated between face feature vectors using cosine_similarity. The closer the score is to 1, the more similar the faces are. If the score exceeds 0.5, the two face features are considered a match.

## Model:
We tried multiple models and eventually selected MobileFaceNet as the face recognition model due to its lightweight nature, making it suitable for embedded systems and mobile devices. It can operate effectively even in resource-limited environments. The model performs well on standard benchmarks like LFW and MegaFace, achieving recognition accuracy close to that of larger models. Furthermore, it utilizes the Linear Bottleneck and Residual Block from the MobileNetV2 architecture, ensuring reasonable performance with reduced computational complexity, faster inference speed, and better real-time face recognition capabilities.
