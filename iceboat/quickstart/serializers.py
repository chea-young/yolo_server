from django.contrib.auth.models import User, Group
from rest_framework import serializers
from quickstart.models import Inference 
from .tasks import run_inference

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')

class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('url', 'name')

class InferenceSerializer(serializers.HyperlinkedModelSerializer):
    class Meta : 
        model = Inference
        fields = ('name','result')
        
    def create(self,validated_data):
        name = validated_data.get('name')
        image = validated_data.get('image')
        inference_obj = Inference.objects.create(name=name,image=image) # or simply Inference.objects.create(**validated_data)
        run_infernece.delay(inference_obj.id) # run inference async        
        return inference_obj
        
        