createsuperuser
ID : iceboat
password : 1234

## 가상 환경 켜기
python3 -m venv env
source venv/Scripts/activate  # On Windows use `env\Scripts\activate`
venv_yolo\Scripts\activate

## restframework 다운
pip install django
pip install djangorestframework

# 파이썬 버전 낮추기
commend 에서 virtualenv 자기 가상환경이른 --python=python버전 하기

# keras 로 darknet 바꾸기 (참고 https://github.com/qqwweee/keras-yolo3)
curl -o yolo.weights https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolov3.h5

python convert.py yolov4.cfg yolov4.weights model_data/yolov4.h5
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

### 푸시알람 + 이미지전송 post 해야되는 순서
- [ ] 1. push 알람 data에 title, body, cctv_id(해당 지역만 알람 보여주기 위해서), image_name사진 이름(현재 시간), 
- [ ] 2. 이미지 저장되는 거 확인(로컬에서)
- [ ] 3. DB에서 이미지 저장되는 거 확인(admin에서)
- [ ] 4. onMessage에 title, option 말고 뭐 더 있는지 확인
- [ ] 5. python 으로 post 이미지 보내는 거 확인
- [ ] 6. 차라리 /start/ url에서 시작하게 만들어서 main이랑 연결하기 (5번 안되면)
    https://my-repo.tistory.com/33
    https://docs.vendhq.com/tutorials/guides/products/image-upload/image_uploads_code_samples/python_requests
- [ ] 7. (전송 됬다는 가정) 받으면 title, body 해주고 if post 데이터가 같은 이름으로 왔을 때  html 파일로 연결해줘서  notification 보내기
     7-1 만약에 notification에 안되면 따로 이미지가 뜨게하기
      7-2 희원오빠 웹앱 완성되는 거 봐서 해당 cctv 쪽에 마크해서 사진올릴 수 있게 하기
