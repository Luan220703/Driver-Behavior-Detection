import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from matplotlib.patches import Rectangle

# Định nghĩa các tên lớp và màu viền tương ứng
class_names = ['Other', 'Safe', 'Talk', 'Text', 'Turn']
border_colors = {'Other': 'orange', 'Safe': 'green', 'Talk': 'red', 'Text': 'red', 'Turn': 'red'}

# Load mô hình đã huấn luyện
model_simplecnn.load_weights('/kaggle/working/model_SimpleCNN.h5')

# Hàm để dự đoán nhãn và thêm viền màu cho khung hình

def predict_and_draw_border(frame, model):
    img = cv2.resize(frame, (Img_width, Img_height))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Dự đoán nhãn
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction, axis=1)[0]]
    
    # Thêm viền màu vào khung hình
    color = border_colors[predicted_class]
    frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)
    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return frame

# Đọc video
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

# Kiểm tra xem video có được mở thành công không
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Đọc và xử lý từng khung hình
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Dự đoán và thêm viền màu
    frame_with_border = predict_and_draw_border(frame, model_simplecnn)
    
    # Hiển thị khung hình
    cv2.imshow('Video', frame_with_border)
    
    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()