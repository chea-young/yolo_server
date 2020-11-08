
from django.db import models 

class Inference(models.Model):
    name = models.CharField(max_length=255)
    result = models.CharField(max_length=255)
    
    def __str__(self):
        return '{name} => result : {result}'.format(name=self.name,result=self.result)

class ImageData(models.Model):
    case = models.CharField(max_length=144)
    image = models.ImageField(upload_to="situation_image",default=False)