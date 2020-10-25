createsuperuser
ID : iceboat
password : 1234

## 가상 환경 켜기
python3 -m venv env
source venv/Scripts/activate  # On Windows use `env\Scripts\activate`

## restframework 다운
pip install django
pip install djangorestframework
# yolo_server


# keras 로 darknet 바꾸기 (참고 https://github.com/qqwweee/keras-yolo3)
curl -o yolo.weights https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
- 이때 convert.py랑 yolov3.cfg 필요

python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]

