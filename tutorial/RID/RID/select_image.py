import os

def select(video_path):
    image_path='./output_image/'
    saved_image=image_path+video_path+'/'
    image_list = os.listdir(saved_image)
    frame_num = 0
    delte_num = 0
    for i in range(len(image_list)):
        frame_name = int(image_list[i][:-4])
        if(frame_num+1 != frame_name and frame_name != 0):
            if(frame_num != 0):
                for j in range(delte_num, frame_num+1):
                    delte_image(saved_image, j)
            frame_num=frame_name
            delte_num = frame_name

def delte_image(image_path, image_name):
    image_file = image_path+image_name+'.png'
    if os.path.isfile(image_file):
        os.remove(image_file)
    
select('accident_video1')