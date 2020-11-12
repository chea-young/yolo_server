
from django.db import models 

class Inference(models.Model):
    name = models.CharField(max_length=255)
    result = models.CharField(max_length=255)
    
    def __str__(self):
        return '{name} => result : {result}'.format(name=self.name,result=self.result)