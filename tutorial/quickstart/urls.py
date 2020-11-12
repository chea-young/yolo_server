from django.urls import path
from rest_framework import routers
from .views import InferenceViewSet
from django.conf.urls import include


router = routers.DefaultRouter()
router.register('inference', InferenceViewSet)

urlpatterns = [
    path('', include(router.urls)),
]