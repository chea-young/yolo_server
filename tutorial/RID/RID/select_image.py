import os

def select(video_path):
    image_list = os.listdir(video_path)
    frame_num = int(image_list[0][:-4])
    delte_num = frame_num

    for i in range(len(image_list)):
        frame_name = int(image_list[i][:-4])
        if(frame_num != frame_name):
            print(frame_num,frame_name)
            for j in range(delte_num+1, frame_num):
                delte_image(video_path, j)
                #delte_num은 알람을 보내고 삭제하기
                #여기 알림보내는 코드 부르기
            frame_num=frame_name
            delte_num = frame_name
        frame_num+= 1
    print(delte_num, frame_num)
    for j in range(delte_num+1, frame_num):
        delte_image(video_path, j)
        #delte_num은 알람을 보내고 삭제하기
        #여기 알림보내는 코드 부르기

def delte_image(image_path, image_name):
    image_file = image_path+str(image_name)+'.png'
    if os.path.isfile(image_file):
        os.remove(image_file)

if __name__ == '__main__':   
    select('accident_video1')