import os
import sys
from scipy.spatial import distance
import time
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import quickstart.RID.RID.core.utils as utils
from quickstart.RID.RID.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from quickstart.RID.RID.core.config import cfg
import requests
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from quickstart.RID.RID.deep_sort import preprocessing, nn_matching
from quickstart.RID.RID.deep_sort.detection import Detection
from quickstart.RID.RID.deep_sort.tracker import Tracker
from quickstart.RID.RID.tools import generate_detections as gdet
from firebase_admin import messaging


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', True, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

START_PATH = 'quickstart/RID/RID/'
#IMAGE_SAVE_PATH = 'C:/Users/w1004/Documents/GitHub/iceboat/tutorial/situation_image/'
IMAGE_SAVE_PATH = 'C:/Users/w1004/Documents/GitHub/iceboat/tutorial/situation_image/'
def saved_log(video_path,bbox_dict,acc_dict,obst_dict,log_num,attracted_image):
    root_path=START_PATH+'/outputs/'
    slash_idx=video_path.rfind('/')
    saved_path=root_path+video_path[slash_idx+1:-4]+'/'
    saved_path_bbox=saved_path+'bbox_log/'
    saved_path_acc=saved_path+'acc_log/'
    saved_path_obst=saved_path+'obstacle_log/'

    # image 저장하는 path
    image_path=START_PATH+'/output_image/'
    saved_image=image_path+video_path[slash_idx+1:-4]+'/'
    saved_image_acc=saved_image+'acc_image/'
    saved_image_obst=saved_image+'obstacle_image/'

    # 영상을 계속 읽고 있는 부부을 찾아서 함수 인자로 받게 하기 그래서 그걸 캡쳐하는 코드를 추가하기
    
    if not os.path.exists(saved_path_bbox):
        os.makedirs(saved_path_bbox)
    if not os.path.exists(saved_path_acc):
        os.makedirs(saved_path_acc)
    if not os.path.exists(saved_path_obst):
        os.makedirs(saved_path_obst)
    
    if not os.path.exists(saved_image_acc):
        os.makedirs(saved_image_acc)
    if not os.path.exists(saved_image_obst):
        os.makedirs(saved_image_obst)

    if bool(bbox_dict)== True:
        with open(saved_path_bbox+str(log_num)+'.txt', "w") as f:
            for key, value in bbox_dict.items():

                f.writelines(f'ID: {key}, BBOX: [ xmin:{value[0]}, ymin:{value[1]}, xmax:{value[2]}, ymax:{value[3]}\n')
        f.close()
    if bool(acc_dict)==True:
        with open(saved_path_acc+str(log_num)+'.txt', "w") as f:
            for key, value in acc_dict.items():

                f.writelines(f'ID: {key}, BBOX: [ xmin:{value[0]}, ymin:{value[1]}, xmax:{value[2]}, ymax:{value[3]}\n')
        f.close()
        print('in',str(log_num))
        cv2.imwrite(saved_image_acc+str(log_num)+'.png',attracted_image)
        # 이미지 선택 하는 코드
        #urls 불러서 이미지 보내는 코드가 django 그래서 이 파일이 부릴

    if bool(obst_dict)==True:
        with open(saved_path_obst+str(log_num)+'.txt', "w") as f:
            for key, value in obst_dict.items():
                f.writelines(f'ID: {key}, BBOX: [ xmin:{value[0]}, ymin:{value[1]}, xmax:{value[2]}, ymax:{value[3]}\n')
        f.close()
        print('in',str(log_num))
        cv2.imwrite(saved_image_obst+str(log_num)+'.png',attracted_image)
        
        
def obstacle_detector(bboxes, background, frame):
    obst_dict=None
    subtracted_result = cv2.absdiff(background, frame)
    subtracted_result_gray = cv2.cvtColor(subtracted_result, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(subtracted_result_gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    positions = np.nonzero(im_bw)

    if positions[0].size == 0:
        return obst_dict, frame
    elif positions[0].max() - positions[0].min() > 136 or positions[1].max() - positions[1].min() > 218:
        return obst_dict, frame
    else:
        obst_dict = {}
        if bboxes.size>0:
            for bbox in bboxes:
                xmin, ymin, width, height  = bbox[0], bbox[1], bbox[2], bbox[3]
                cv2.rectangle(frame, (xmin, ymin), ( xmin + width, ymin+height), (255,255,255), -1) 

        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        right = positions[1].max()
        #print('top:', top, 'bottom:', bottom, 'left:', left, 'right:', right)
        obst_dict['obstacle_location']=[int(left),int(top),int(right),int(bottom)]
        resulted_frame = cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 3)
        cv2.putText(resulted_frame, 'Obstacle', (right, top), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
        return obst_dict, resulted_frame

def main(_argv):

    input_dir=START_PATH+'/video/'
    output_dir=START_PATH+'/outputs/'
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_list = os.listdir(input_dir)

    if len(video_list)==0:
        raise ValueError('Empty files in video folder')

    for i in range(len(video_list)):
        #initialization for accident threshold and log
        acc_threshold= 0.15
        acc_log=False

        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        
        # initialize deep sort
        model_filename = START_PATH+'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        NUM_CLASS=1

        FLAGS.video=input_dir+video_list[i]
        FLAGS.output=output_dir+video_list[i][:-3]+'avi'
        input_size = FLAGS.size
        video_path = FLAGS.video
        # load tflite model if flag is set
        if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=START_PATH+FLAGS.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load(START_PATH+FLAGS.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

        # begin video capture
        #cv2.VideoCapture -> 영상을 받아오는 부분
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        out = None

        # get video ready to save locally if flag is set
        send_frame = 0
        frame_num = 0
        if FLAGS.output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

        # while video is running
        log_num=0
        background=None
        while True:
            # 비디오가 재생되면 한 프레임씩 읽는다.
            return_value, frame = vid.read()
            image_value, attracted_image = vid.read()

            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('------------------------------')
                print('Video has ended or failed, try a different video format!', str(log_num))
                break

            if log_num==0:
                background=frame
        
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            # run detections on tflite if flag is set
            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to customize tracker for only people)
            #allowed_classes = ['person']

            #obstacle detection model 
            obst_dict, frame = obstacle_detector(bboxes, background, frame)

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            #attracted_image = frame

            if FLAGS.count:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                print("Objects being tracked: {}".format(count))
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks and save tracking information
            tracker_dict={}
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                
            # draw bbox on screen
                tracker_dict[str(track.track_id)]=[int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]

                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                #cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            acc_dict={}

            
            # find overlapping area between cars
            for f1 in range(len(tracker_dict.keys())-1):
                s_id= list(tracker_dict.keys())[f1]

                s_xmin,s_ymin,s_xmax, s_ymax= tracker_dict[str(s_id)]
                for f2 in range(f1+1, len(tracker_dict.keys())):

                    c_id= list(tracker_dict.keys())[f2]
                    c_xmin,c_ymin,c_xmax, c_ymax= tracker_dict[str(c_id)]
                    
                    if s_xmax < c_xmin:
                        continue
                    if s_xmin > c_xmax:
                        continue
                    if s_ymax < c_ymin:
                        continue
                    if s_ymin > c_ymax:
                        continue
                    
                    left_up_x = max(s_xmin, c_xmin)
                    left_up_y = max(s_ymin, c_ymin)
                    right_down_x = min(s_xmax, c_xmax)
                    right_down_y = min(s_ymax, c_ymax)
                    
                    width = right_down_x - left_up_x
                    height =  right_down_y - left_up_y
                    
                    s_area= (s_xmax- s_xmin) *(s_ymax-s_ymin )
                    c_area= (c_xmax- c_xmin) *(c_ymax-c_ymin )
                    overlap_area= width*height

                    if s_area > c_area:
                        criteria_area= c_area
                    else:
                        criteria_area=s_area
                    
                    if (overlap_area/criteria_area ) > acc_threshold:
                        acc_log=True
                    
                    cv2.rectangle(frame, (int(left_up_x), int(left_up_y)), (int(right_down_x), int(right_down_y)), (255,255,255), 2)
                    cv2.putText(frame,  "overlap-" + str(s_id)+'and'+str(c_id),(int(left_up_x), int(left_up_y-10)),0, 0.75, (255,0,0),2)
                    acc_id=str(s_id)+'and'+str(c_id)
                    acc_dict[acc_id]=[int(left_up_x),int(left_up_y),int(right_down_x),int(right_down_y)]
            
            # accident alarm
            if acc_log ==True:
                cv2.rectangle(frame, (int(0), int(0)), (int(450), int(70)), (255,4,0), 4)
                cv2.putText(frame,  "accident occured",(int(10), int(50)),0, 1.5, (255,0,0),4)        
            
            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if not FLAGS.dont_show:
                cv2.imshow("Output Video", result)
            if not os.path.exists(IMAGE_SAVE_PATH):
                os.makedirs(IMAGE_SAVE_PATH)
            tm = time.localtime()
            image_name = str(tm.tm_mon)+str(tm.tm_mon)+str(tm.tm_hour)+str(tm.tm_min)+str(tm.tm_sec)
            if bool(acc_dict)==True:#사고난 것을 감지하면
                if(frame_num == 0):
                    frame_num = log_num
                    image_url = IMAGE_SAVE_PATH+image_name+'.png'
                    cv2.imwrite(image_url,attracted_image) #local 저장소에 감지됬을 때 이미지를 저장
                    #'acc'는 감지 된 유형의 정보, image_name은 저장된 이미지 이름으로 fcm을 보낸다.
                    send_to_firebase_cloud_messaging('acc', image_name) 
                    #image_url은 local에 저장된 이미지 장소, image_name 이미지로 post 보냄
                    #image_name을 보내는 이유는 fcm을 보낼 떄 image_name이 같은 것으로 DB에서 이미지를 찾아서 알림을 보내기 때문에
                    send_post(image_url, image_name)
                else:
                    if(frame_num+1 != log_num):
                        image_url = IMAGE_SAVE_PATH+image_name+'.png'
                        cv2.imwrite(image_url,attracted_image)
                        send_to_firebase_cloud_messaging('acc', image_name)
                        send_post(image_url, image_name)
                        frame_num = log_num
                    else:
                        frame_num += 1
            elif(bool(obst_dict)==True):
                if(frame_num == 0):
                    frame_num = log_num
                    image_url = IMAGE_SAVE_PATH+image_name+'.png'
                    cv2.imwrite(image_url,attracted_image)
                    send_to_firebase_cloud_messaging('ob',image_name)
                    send_post(image_url, image_name)
                else:
                    if(frame_num+1 != log_num):
                        image_url = IMAGE_SAVE_PATH+image_name+'.png'
                        cv2.imwrite(image_url,attracted_image)
                        send_to_firebase_cloud_messaging('ob',image_name)
                        send_post(image_url, image_name)
                        frame_num = log_num
                    else:
                        frame_num += 1

            saved_log(video_path,tracker_dict,acc_dict,obst_dict,log_num,attracted_image)
            # if output flag is set, save video file
            if FLAGS.output:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            log_num+=1  

        cv2.destroyAllWindows()
#fcm 보내는 함수
def send_to_firebase_cloud_messaging(type, image_name):
    registration_token = 'dEn7h6FyDEw:APA91bFvJa1wBNZyCwd7d5wkwrAuFm_Dh-ryAlSi0UO-f_dntQEK5G2W6vsqIBtz_QVwRdWJlf6__YFk48P6tJuXmx6-LW8vgoEefdhPpDvhQ6qnW12Sz8vLNdG49Fws9JakpSoa0B5u'
    if type == 'acc' :
        message = messaging.Message(
            data={
                'title':'조심하세요',
                'body':'경로에 교통사고가 났습니다. 주의하세요', 
                'cctv_id' : '1',
                'image_name' : image_name,
            },
            token=registration_token,
        )
        response = messaging.send(message)
        print('Successfully sent message:', response)
    elif type == 'ob' :
        message = messaging.Message(
            data={
                'title':'조심하세요',
                'body':'경로의 도로에 장애물이 있습니다. 주의하세요', 
                'cctv_id' : '1',
                'image_name' : image_name,
            },
            token=registration_token,
        )
        response = messaging.send(message)
        print('Successfully sent message:', response)
#post로 보내는 함수
def send_post(image_url, image_name):
    url = "http://127.0.0.1:8000/push_server/image/"
    files = {
        'image': open(image_url, 'rb')
        }
    data = {
        'name' : image_name
    }
    response = requests.request("POST", url, files=files, data = data)
    print(response)
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
else:
    app.run(main)
