from django.urls import path
from rest_framework import routers
from .views import InferenceViewSet, DetectStartView, IndexView
from django.conf.urls import include

app_name = 'quickstart'

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('detect_start/', DetectStartView.as_view())
]