# 📸 FaceDApps
Application for facial expression detection

When it comes to facial expression identification, the Fer2013 dataset is a popular benchmark. Over 35,000 grayscale photos that have been divided into seven emotional categories: disgust, anger, fear, happy, sad, surprise, and neutral, make up the dataset. Since each image is assigned a label corresponding to one of these emotion classes, it may be used to train and assess emotion recognition algorithms. The breadth of this dataset and the range of facial expressions recorded in various backgrounds and lighting situations make it a popular choice for deep learning model training.

## 📦 Dataset
In developing this final project, the goal is to identify the type of facial expression through images. The FER2013 Images Afaq (2023) dataset taken from the Kaggle platform is used to collect the required data. The images are equipped with category labels for each class. In this study, the images used are publicly available data. The data comes from the Facial Expression Recognition dataset collected from the Kaggle site.

## ⚙️ Stages 
**1. Data Collection:** 
The dataset applied in this project is taken from Kaggle and includes 35 thousand images of facial expressions categorized as angry, disgusted, fearful, happy, sad, surprised, and neutral.

**2. Data Preprocessing:**
In order for the model to learn images effectively, preprocessing steps are applied. In this case, face cropping is applied to extract only the facial part of the image. The goal is to lighten the computational load and prevent the model from processing irrelevant information from the image.

🗂️ Dataset Division
- Three categories of data are applied in this study: training data, validation data, and testing data. Training data is used to build a model that is able to recognize facial images and classify them based on the features that have been obtained. Validation data serves to assess the performance of the model during the training process, while testing data is used to assess the effectiveness of the model in classifying data that has not been used previously in training or validation with a ratio of 80:10:10.

🔄 Data Augmentation
- Cropping or image cutting, which is removing unnecessary parts of the image, so that only the face image remains.
- Rescale or resizing, which is an enhancement method that modifies the pixel size of the image, is generally used for normalization.
- Flip or inversion, which is a data enhancement technique that reverses the position of pixels in an image both horizontally and vertically.
- Zoom or enlargement, is an enhancement technique that is useful for enlarging (zooming in) or reducing (zooming out) images within specified limits.
- Median Filter is applied to reduce noise in the image while maintaining important details.
- Grayscale is used to improve color quality, where facial images with red, green, and blue (RGB) components will be converted into grayscale images.
- SMOTE is used to ensure that the model gets quality and even training data, so that it can learn effectively and produce accurate predictions.

**3. Data Modelling:**
VGG16 Pretrained Model
- A convolutional neural network (CNN) architecture pre-trained on the large ImageNet dataset. The model was developed by the Visual Geometry Group (VGG)
Oxford and consists of 16 convolutional layers with 3x3 filters, followed by a fully
connected layer. Because it is pre-trained, the model can be used directly for image classification or adapted through transfer learning (fine-tuning) for specific tasks with limited datasets, saving time and computation compared to training from scratch.

**4. Model Training :**
The model is trained using a facial expression dataset that is divided into training and validation data. To prevent overfitting, EarlyStopping is used which will stop training if there is no increase in accuracy or decrease in loss for 5 consecutive epochs. 

**5. Model Testing :**
The model testing stage aims to assess how effective the CNN model is
using the vgg16 architecture in recognizing various types of facial expressions accurately. The assessment is carried out by utilizing the validation data set

**6. Result and Evaluation :**
- 🏆 Best Training Accuracy    : 0.9527
- 🏆 Best Validation Accuracy  : 0.6723
- 🎯 Final Training Accuracy   : 0.9515
- 🎯 Final Validation Accuracy : 0.6667
- 🧪 Test Accuracy             : 0.5651

## 🖥️ Deployment
To deploy it, use ngrok so that it can be opened publicly.
A demo is temporarily available via ngrok

🚀 How to Run
1. Clone the repository:
  ``` bash
  https://github.com/fikrimunadi1/FaceDApps.git
  cd FaceDApps
  ```
2. Download flask 
3. Install requirement.txt
  ``` bash
   pip install -r requirements.txt
  ```
5. Run Flask backend in terminal Visual Studio Code
  ``` bash
   python app.py
  ``` 
6.Open in browser or chrome
  ``` bash
   http://localhost:5000/ 
  ``` 
