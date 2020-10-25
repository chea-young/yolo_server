from django.contrib import admin
from django.urls import include, path
from rest_framework import routers
from django.conf.urls.static import static
from django.conf import settings
from quickstart import views 	#이 부분 수정함

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)
router.register(r'inference', views.InferenceViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('admin/', admin.site.urls),
    path('api/',include('quickstart.urls')),
]