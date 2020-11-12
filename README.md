createsuperuser
ID : iceboat
password : 1234

## 가상 환경 켜기
python3 -m venv env
source venv/Scripts/activate  # On Windows use `env\Scripts\activate`

## restframework 다운
pip install django
pip install djangorestframework

# 파이썬 버전 낮추기
commend 에서 virtualenv 자기 가상환경이른 --python=python버전 하기

# keras 로 darknet 바꾸기 (참고 https://github.com/qqwweee/keras-yolo3)
curl -o yolo.weights https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
- 이때 convert.py랑 yolov3.cfg 필요

python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]

yolo_anchor -  임이의 박스 크기 (여기서 정한)

python yolo_video.py --image : 사진 입력을 받음 -> BMP으로 결과 나옴 -> .save 저장하게 만듦 -> result에 결과 저장 (날짜, 시간순으로)
- yolo.h5 model, anchors and classes loaded 필요

## yolo 서버의 restframework 
https://medium.com/@chamakhabdallah8/how-to-deploy-a-keras-model-to-production-with-django-drf-celery-and-redis-df4901014355 


### 이미지 전송 restful
- https://yongwookha.github.io/ETC/2020-07-22-django-rest-api-framework
- https://eunjin3786.tistory.com/133
- https://itinerant.tistory.com/134?category=736038 => basic.html
- https://bourbonkk.tistory.com/69 => send_post.html

### js 실행시키기
- https://iamaman.tistory.com/2058

### 이미지 전송 시나리오
- python manage.py runserver 127.0.0.1:8008 실행
- http://127.0.0.1:8008/quickstart/detect_signal/ 화면을 치면
- push_server로 http://127.0.0.1:8000/get_image/ post로 보내지고 자동으로 DB에 저장

- (index):1 Access to XMLHttpRequest at 'http://127.0.0.1:8000/quickstart/sample/' from origin 'http://127.0.0.1:8008' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
--> 해결을 위해 $ pip install django-cors-headers

### 장고 시작시 실행할 코드 작성
- http://blog.quantylab.com/django_onstartup.html -> python manage.py runserver --noreload
