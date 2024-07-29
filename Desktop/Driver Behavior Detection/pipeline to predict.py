import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

def simple_cnn(img_width, img_height, num_classes):
    input_layer = Input(shape=(img_width, img_height, 3))
    
    x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

num_classes = 5
img_width, img_height = 240, 240
model_simplecnn = simple_cnn(img_width, img_height, num_classes)

model_simplecnn.load_weights('/kaggle/working/model_SimpleCNN.h5')

class_names = ['Other', 'Safe Driving', 'Talking Phone', 'Texting Phone', 'Turning']

import random
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Định nghĩa các tên lớp và màu viền tương ứng
class_names = ['Other', 'Safe', 'Talk', 'Text', 'Turn']
border_colors = {'Other': 'orange', 'Safe': 'green', 'Talk': 'red', 'Text': 'red', 'Turn': 'red'}

# Chọn ngẫu nhiên 20 tấm ảnh từ tập test
sample_images = random.sample(list(test_df['image']), 20)

# Vẽ các tấm ảnh và dự đoán nhãn của chúng
plt.figure(figsize=(20, 20))
for i, img_path in enumerate(sample_images):
    # Load và xử lý ảnh
    img = load_img(img_path, target_size=(Img_height, Img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Dự đoán nhãn
    prediction = model_simplecnn.predict(img_array)
    predicted_class = class_names[np.argmax(prediction, axis=1)[0]]
    
    # Hiển thị ảnh và viền màu
    plt.subplot(5, 4, i + 1)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    
    # Thêm viền màu dựa trên nhãn
    ax = plt.gca()
    rect = patches.Rectangle((0, 0), Img_width, Img_height, linewidth=5, edgecolor=border_colors[predicted_class], facecolor='none')
    ax.add_patch(rect)

plt.tight_layout()
plt.show()