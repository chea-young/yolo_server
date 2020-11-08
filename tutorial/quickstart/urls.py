from django.urls import path
from rest_framework import routers
from . import views
from django.conf.urls import include

app_name = 'quickstart'

urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('detect_start/', views.DetectStartView.as_view()),
    path('detect_signal/', views.detect_situation)
]