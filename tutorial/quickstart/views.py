from django.shortcuts import render
from django.contrib.auth.models import User, Group
#from rest_framework.decorators import action
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import viewsets,status,generics
from .serializers import UserSerializer, GroupSerializer
from .serializers import InferenceSerializer
from .models import Inference, ImageData
from celery.result import AsyncResult
#from RID.RID.main import run
from django.views import View
from django.http import HttpResponse, JsonResponse
import cv2
import os


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer

class GroupViewSet(viewsets.ModelViewSet):
    queryset = Group.objects.all()
    serializer_class = GroupSerializer

class InferenceViewSet(viewsets.ModelViewSet):
    serializer_class = InferenceSerializer
    queryset = Inference.objects.all()
    
    @api_view(['GET'])   
    def monitor_inference_progress(self,request,slug):
        inference_obj = self.get_object()
        progress = 100
        result  = AsyncResult(inference_obj.task_id)
        if isinstance(result.info, dict):
            progress  = result.info['progress']
        description = result.state
        return Response({'progress':progress,'description':description},status=status.HTTP_200_OK)
class IndexView(View):
    def get(self, request):
        dummy_data = {
            'name': '죠르디',
            'type': '공룡',
            'job': '편의점알바생',
            'age': 5
        }
        return JsonResponse(dummy_data)

    def post(self, request):
        return HttpResponse("Post 요청을 잘받았다")

    def put(self, request):
        return HttpResponse("Put 요청을 잘받았다")

    def delete(self, request):
        return HttpResponse("Delete 요청을 잘받았다")

#push_serer Check로부터 알림
class DetectStartView(View):
    def get(self, request):
        dummy_data = {
            'name': '죠르디',
            'type': '공룡',
            'job': '편의점알바생',
            'age': 5
        }
        return JsonResponse(dummy_data)

    def post(self, request):
        return HttpResponse("Post 요청을 잘받았다")

    def put(self, request):
        return HttpResponse("Put 요청을 잘받았다")

    def delete(self, request):
        return HttpResponse("Delete 요청을 잘받았다")

import js2py
# 동영상을 주기적으로 계속 감시, init으로 넣어야 되나..
#select는 주기적으로 
def detect_situation(request):
    #js2py.translate_file('send.js', 'send.py')
    #select(video_path, image)
    # send_image(path, image)
    print('send')
    return render(request, 'quickstart/basic.html')

def select(video_path,case):
    image_list = os.listdir(video_path)
    frame_num = int(image_list[0][:-4])
    delte_num = frame_num

    for i in range(len(image_list)):
        frame_name = int(image_list[i][:-4])
        if(frame_num != frame_name):
            print(frame_num,frame_name)
            for j in range(delte_num+1, frame_num):
                delte_image(video_path, j)
            image = cv2.imread(video_path+str(delte_num)+'.png')
            form = ImageData(case=case, image = image)
            form.save()
            detect_situation()#여기 알림보내는 코드 부르기
            delte_image(video_path, delte_num)#delte_num은 알람을 보내고 삭제하기
            frame_num=frame_name
            delte_num = frame_name
        frame_num+= 1
    print(delte_num, frame_num)
    for j in range(delte_num+1, frame_num):
        delte_image(video_path, j)
    image = cv2.imread(video_path+str(delte_num)+'.png')
    form = ImageData(case=case, image = image)
    form.save()
    detect_situation()#여기 알림보내는 코드 부르기
    delte_image(video_path, delte_num)#delte_num은 알람을 보내고 삭제하기

def delte_image(image_path, image_name):
    image_file = image_path+str(image_name)+'.png'
    if os.path.isfile(image_file):
        os.remove(image_file)