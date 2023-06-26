# WBC Differential Learning Tool

Human peripheral blood contains 3 main types of cells, red blood cells, white blood cells and platelets. In normal clinical laboratory routine tests, we usually further differentiate white blood cells into 5 main different classes: **neutrophil, lymphocyte, basophil, eosinophil and monocyte**. 

The reason why we would like to classify those cells is because every class should have a set percentage of distribution in the normal human body. The sudden change in percentage of each cell class gives physicians an idea of what went wrong in the body. For example, abnormally high basophils and lymphocytes can mean you have chronic infection, cancer or other autoimmune disease. Sudden abnormal increase of eosinophil might mean your body is fighting for parasitic infection. Since the classification of these white blood cells affect the further treatment decision, it is very important that we correctly classify them.

The purpose of this module is for educating medical school students or newly trained clinical laboratory scientists. This application allow them to see the unique characteristic of each class of cells and further train them to correctly classify different cell types.

## Data
The dataset that I use for this module is from a free public resource from Raabin-WBC data
[Free Data - Raabin Health Database](https://raabindata.com/free-data/#acute-lymphoblastic-leukemia).

As I went through the dataset, I’ve noticed some of the pictures contain cells from different classes in the same picture, but they’ve only labeled the most centered cell. Therefore, I manually relabel them using [Roboflow](https://universe.roboflow.com/duke-aipi-540-summer-2023/wbc-classification-ih8we/model/1) so that all of the main cells were labelled. The re-labelled dataset became a useful training set for the YOLOv5 model.

![Screenshot](https://github.com/changyuhsin1999/WBC-Differential-Learning-Tool/blob/main/Image/Screen%20Shot%202023-06-18%20at%201.57.33%20AM.png)

## Model
In this computer vision module, I used 3 different models to experiment with the performance of classifying different white blood cells, including the classic machine learning SVC model, neural network Resnet 34 and YOLOv5. Below is a brief comparison of each of the models with different evaluation methods.

### Resnet and SVC
| Model         | Accuracy      |
| ------------- |:-------------:|
| Resnet34      | 97.1 %        |
| SVC           | 10.9 %        |

### YOLOv5
mAP | Precision | Recall
--- | --- | ---
97.4 % | 95.4 % | 95.7 %

## Discussion
Using the Resnet34 model, I observed that the model was particularly struggle with classifying lymphocytes with recall of 0.882, which correlate with human eye training since lymphocytes and monocytes are the 2 main classes that we commonly misclassify.

## How to get started
### Prepare your environment

```
conda create --name envir python=3.7.15
conda activate envir
```
### Install requirement.txt

```
pip install -r requirements.txt
```
### Train the model
Run the resnet.py and SVC.py python script to train and evaluate your model

## Training YOLOv5 model
### Clone repo
clone repo from the well established pre-trained framework for transfer learning: [utralytics](https://github.com/ultralytics/yolov5)
### Install requirement
```
pip install -U -r yolov5/requirements.txt
```
### Modify the yolov5s.yaml file under models
Change nc (number of class) to 7 for training WBC Roboflow dataset and save the file
### Run YOLOv5.py for training

## Streamlit Demo
### Create account on Roboflow
obtain API key from [Roboflow](https://universe.roboflow.com/duke-aipi-540-summer-2023/wbc-classification-ih8we/model/1)
### Replace API_key 
replace api_key = [Your API key] in the streamlit.py file
### Run the application
```
streamlit run streamlit.py
```

## Reference
Previous research: [Raabin_health_database](https://www.nimaadmed.com/raabin_health_database/)
Streamlit application framework/tutorial: [Streamlit_blog](https://blog.streamlit.io/how-to-use-roboflow-and-streamlit-to-visualize-object-detection-output/)
Yolov5 Transfer Learning repo: [utralytics](https://github.com/ultralytics/yolov5)
