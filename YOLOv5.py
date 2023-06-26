%cd /content/yolov5
%cd /content/

#Unzip the dataset obtain from Roboflow
curl -L "https://app.roboflow.com/ds/gP4GxvSn8q?key=vapjRAJzcV" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

python yolov5/train.py --img 224 --batch 8 --epochs 20 --data data.yaml --cfg yolov5s.yaml --weights yolov5s.pt --cache
