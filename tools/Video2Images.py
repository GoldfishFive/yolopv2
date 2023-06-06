import cv2, os


def get_rainy_image():
    root = "/home/data/highway/0095/rainy/"
    l = ['000055.ts', '000056.ts', '000057.ts', '000059.ts', '000060.ts', '000063.ts', ]
    # for i in os.listdir(root):
    for i in l:
        video_path = os.path.join(root, i)
        cap = cv2.VideoCapture(video_path)
        save_path = "/home/data/highway/0095/rainy_image_list/" + i
        os.makedirs(save_path, exist_ok=True)

        c = 0
        rval = cap.isOpened()
        k = 0
        while rval:
            c = c + 1
            rval, frame = cap.read()
            if rval and c % 10 == 0:
                # cv2.imwrite(os.path.join(save_path, str(k).zfill(6) + '.jpg'), frame)
                cv2.imwrite(os.path.join(save_path, i + "_" + str(k).zfill(6) + '.jpg'), frame)
                print(video_path, c, k)
                k += 1
            elif rval:
                pass
            else:
                break


def get_foggy_image():
    root = "/home/data/highway/0095/foggy/"
    l = ['000004.ts', '000005.ts', '000062.ts', '000081.mp4']
    # for i in os.listdir(root):
    for i in l:
        video_path = os.path.join(root, i)
        cap = cv2.VideoCapture(video_path)
        save_path = "/home/data/highway/0095/foggy_image_list/"
        os.makedirs(save_path, exist_ok=True)

        c = 0
        rval = cap.isOpened()
        k = 0
        while rval:
            c = c + 1
            rval, frame = cap.read()
            if rval and c % 100 == 0:
                # cv2.imwrite(os.path.join(save_path, str(k).zfill(6) + '.jpg'), frame)
                cv2.imwrite(os.path.join(save_path, i + "_" + str(k).zfill(6) + '.jpg'), frame)
                print(video_path, c, k)
                k += 1
            elif rval:
                pass
            else:
                break


def samples_video2images_120():
    '''
    平均每段视频截取120帧，不考虑视野场景切换
    '''
    root = r"/home/data/highway/0095/"
    dir = ['foggy', 'rainy', 'snowy', 'sunny']
    save_path = r"/home/data/highway/0085_images"

    for i in dir:
        video_dir = os.path.join(root, i)
        for j in os.listdir(video_dir):
            vc = cv2.VideoCapture(os.path.join(video_dir, j))

            print(os.path.join(video_dir, j), vc.get(7), vc.get(7) / vc.get(5))  # 视频文件中的帧数,帧数/帧率
            # os.makedirs(os.path.join(save_path, j), exist_ok=True)

            c = 0
            k = 0
            rval = vc.isOpened()
            interval = int(vc.get(7) / 120)
            while rval:
                c = c + 1
                rval, frame = vc.read()
                if rval and c % interval == 0:
                    cv2.imwrite(os.path.join(save_path, j + "_" + str(k).zfill(6) + '.jpg'), frame)
                    print(os.path.join(video_dir, j), c, k)
                    k += 1
                elif rval:
                    pass
                else:
                    break


def getLen():
    root = '/home/data/highway/0085_images/'
    print(len(os.listdir(root)))


def samples_video2images():
    root = "/home/user01/datasets/traffic/Highway_sample/"
    save_root = "/home/user01/datasets/traffic/Highway_sample_images"

    for i in os.listdir(root):
        video_dir = os.path.join(root, i)
        for j in os.listdir(video_dir):

            vc = cv2.VideoCapture(os.path.join(video_dir, j))
            # print(os.path.join(video_dir, j), vc.get(7), vc.get(7)/30) #视频文件中的帧数
            # print(vc.get(7)/30,'\n') #视频文件中的帧数
            save_path = os.path.join(save_root, j)
            os.makedirs(save_path, exist_ok=True)

            c = 0
            rval = vc.isOpened()
            k = 0
            while rval:
                c = c + 1
                rval, frame = vc.read()
                # if rval and c % 30 == 0:
                if rval:
                    # cv2.imwrite(os.path.join(save_path, str(k).zfill(6) + '.jpg'), frame)
                    cv2.imwrite(os.path.join(save_path, j + "_" + str(k).zfill(6) + '.jpg'), frame)
                    print(os.path.join(video_dir, j), c, k)
                    k += 1
                elif rval:
                    pass
                else:
                    break


def samples_images2video():
    # change your path; 30 is fps; (2304,1296) is screen size
    videoWriter = cv2.VideoWriter('./video/savevideo/test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (2304, 1296))

    for i in range(1, 1684):
        # load pictures from your path
        img = cv2.imread('./processedpic/8/' + 'processed-' + str(i) + '.png')
        img = cv2.resize(img, (2304, 1296))
        videoWriter.write(img)

    videoWriter.release()


def get_sunny_images():
    root = "/home/data/highway/0095/sunny/"
    l = ['000072.ts', '000021.mp4', '000011.mp4', '000089.mp4']
    # for i in os.listdir(root):
    for i in l:
        video_path = os.path.join(root, i)
        cap = cv2.VideoCapture(video_path)
        # save_path = "/home/data/highway/0095/sunny_image_list/" + i
        save_path = "/home/data/highway/0095/sunny_image_list/" # 放一起
        os.makedirs(save_path, exist_ok=True)

        c = 0
        rval = cap.isOpened()
        k = 0
        while rval:
            c = c + 1
            rval, frame = cap.read()
            if rval and c % 10 == 0:
                # cv2.imwrite(os.path.join(save_path, str(k).zfill(6) + '.jpg'), frame)
                cv2.imwrite(os.path.join(save_path, i + "_" + str(k).zfill(6) + '.jpg'), frame)
                print(video_path, c, k)
                k += 1
            elif rval:
                pass
            else:
                break
def testyacs():
    from yacs.config import CfgNode as CN
    yaml_name='../lib/config/YOLOPv2.yaml'
    fcfg = open(yaml_name)
    cfg = CN.load_cfg(fcfg)


if __name__ == "__main__":
    # samples_video2images()
    # get_foggy_image()

    # samples_video2images_120()
    # getLen()
    # get_rainy_image()
    # d = '/home/data/highway/0095/rainy_image_list/'
    # for i in os.listdir(d):
    #     print("python test_PReNet.py --logdir logs/Rain100H/PReNet6 \
    #      --save_path results/PReNet/{} --data_path /home/data/highway/0095/rainy_image_list/{}".format(i,i))

    # get_sunny_images()
    testyacs()
    pass
