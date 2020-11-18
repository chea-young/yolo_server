from tutorial.celery import app
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from .models import Inference


@app.task(bind=True)
def run_inference(self, inference_id):

    # update the task id on inference_obj the for future monitoring
    inference_obj = Inference.objects.get(id=inference_id)
    inference_obj.task_id = self.request.id
    inference_obj.save()
    
    # image preprocessing
    self.update_state(state='image preprocessing', meta={'progress': 0})   
    img = image.load_img(inference_obj.image.file.filename,target_size=(150,150))
    img_array = image.img_to_array(img)
    img_array.shape = (1,150,150,3)
    
    self.update_state(state='load the model', meta={'progress': 25})
    # load model architecture
    with open('../yolov3.cfg', 'r') as f:
        model = model_from_json(f.read())
    # load model weights
    model.load_weights('../model_data/yolo.h5')
    
    # predict the image
    self.update_state(state='predict the image', meta={'progress': 50})
    prediction = model.predict(img_array,verbose=1) 
    result = 'unkown'
    if prediction < 0.5 :
        result = 'Normal'
    elif prediction > 0.5 :
        result = 'Pneumonia'
        
    # save sesult to database
    inference_obj.result = result
    inference_obj.save()
    
    self.update_state(state='Finished', meta={'progress': 100})