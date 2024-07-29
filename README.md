# Driver-Behavior-Detection
According to data from the Vietnam Traffic Police Department, each year Vietnam records approximately 15,000 to 20,000 traffic accidents, resulting in about 7,000 to 8,000 deaths and tens of thousands of injuries. One of the primary causes of these accidents is unsafe driving behaviors, such as drowsiness and loss of concentration while driving, etc.

Driver behavior detection uses technology to monitor and analyze driver behavior to improve traffic safety. Technologies such as sensors, cameras, GPS, machine learning algorithms, and deep learning models are used to identify dangerous behaviors such as distraction, drowsiness, or speeding. This application helps reduce accidents and supports a safe driving system.

To solve the above problem, Group 3 decided to use CNN (an architecture of Neural Network) models based on the TensorFlow library including models such as AlexNet, VGG, and Resnet... to predict and monitor driving behaviors based on optimizing actual accuracy

 # Dataset
 The dataset is from Kaggle of Author “ROBINRENI’
 (https://www.kaggle.com/datasets/robinreni/revitsone-5class)
This dataset is about 500 MB, including 10,766 photos (JPG & PNG)
This is a dataset collected for a fleet management project. This dataset contains 5 classes
-Safe Driving 
-Talking Phone
-Texting Phone
-Turning
-Other Activities
# Augumentation & Input Model
1. Augumentation
   ![image](https://github.com/user-attachments/assets/8a09af29-543d-4ea9-a594-66422a9d30c4)
2. Input Model
   ![image](https://github.com/user-attachments/assets/a1c29d79-c4dc-4afc-a4f6-6db40d15a28e)


# Pipeline of Team 
![image](https://github.com/user-attachments/assets/f44b36fc-5b35-41fd-9221-a1e7110bd94c)

# Result 
Before Augumentation
![image](https://github.com/user-attachments/assets/8b3f2465-c64a-40b1-beec-e410fae6bb5b)
After Augumentation
![image](https://github.com/user-attachments/assets/bccf7605-bb61-4169-b587-1264792399c4)

# About THE loss 
Because this dataset has many images with overlapping labels like texting phone images, maybe some images are similar but they are talking phones, so it also becomes a multi-label classification problem, so using Binary Classification has high results.
![image](https://github.com/user-attachments/assets/e7a7420d-e9bf-47f4-9582-05f2e123021e)



