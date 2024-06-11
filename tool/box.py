# -*- coding: utf-8 -*-
# @Time    : 2022/7/29 9:45
# @Author  : Fanstandy
# @CoAthor : Xu
# @Updating Time : 2023/2/8 17:33
import cv2
from PIL import Image


def auto_box(img, left1, w1, h1, file_name, save_path , left2 = None, w2 = None, h2 =None):
    color = (0, 0, 255)  # 框颜色 BGR
    width = 1   # 线宽度
    right1 = []
    right1.append(left1[0]+w1)
    right1.append(left1[1]+h1)
    cut_img = img[left1[1]:right1[1], left1[0]:right1[0]]
    cut_img = img[left1[1]:left1[1]+h1, left1[0]:left1[0] + w1]
    box_path = save_path + file_name + '_box1' + '.PNG'
    img_path = save_path + file_name + '.PNG'
    cv2.imwrite(box_path, cut_img)
    cv2.rectangle(img, (left1[0], left1[1]), (right1[0], right1[1]), color, width)  # 画框
    cv2.rectangle(img, (left1[0], left1[1]), (left1[0] + w1, left1[1]+h1), color, width)  # 画框
    if left2 == None or w2 == None or h2 == None:
        cv2.imwrite(img_path, img)
    else:
        color = (0, 255, 0)  # 框颜色 BGR
        right2 = []
        right2.append(left2[0]+w2)
        right2.append(left2[1]+h2)
        cut_img = img[left2[1]:right2[1], left2[0]:right2[0]]
        cut_img = img[left2[1]:left2[1]+h2, left2[0]:left2[0] + w2]
        box_path = save_path + file_name + '_box2' + '.PNG'
        cv2.imwrite(box_path, cut_img)
        cv2.rectangle(img, (left2[0], left2[1]), (right2[0], right2[1]), color, width)  # 画框
        cv2.rectangle(img, (left2[0], left2[1]), (left2[0] + w2, left2[1]+h2), color, width)  # 画框
        cv2.imwrite(img_path, img)

def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        print('左上角坐标{},{}'.format(x, y))
        cv2.circle(img2, point1, 10, (0, 255, 0), 1)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 1)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        print('右下角坐标{},{}'.format(x, y))
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 1)
        cv2.imshow('image', img2)
        # cv2.imwrite('new.png', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        print('width:{},height:{}'.format(width, height))
        cut_img = img[min_y:min_y + height, min_x:min_x + width]
        cv2.imshow('area', cut_img)


def get_left(img):
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)  # 按任意键结束

if __name__ == '__main__':
    # main()
    global img
    global point1, point2
    # path = 'E:/RTNIF/paper/compare/TNO/Ours/Two_stage/AFBR/'
    # path = 'E:/RTNIF/paper/compare/TNO/Ours/Autocoder_100/'
    left1 = [55,160]    # 左上角坐标（x,y）
    w1 = 45     #宽
    h1 = 65      #高
    left2 = [145,170]    # 左上角坐标（x,y） // left = None or [x,y]
    w2 = 50     #宽     #None or int
    h2 = 60     #高     #None or int
    bf = 0
    if bf == 0:
        path = 'E:/RTNIF/paper/compare/RoadScence/state_of_arts/'
        file_name = ['UMF-CMGR','MSPIF']
        for name in file_name:
            img = cv2.imread(path + name + '/90.png')     #img_path
            bz = 0
            save_path = 'box/'  #保存路径
            if bz == 1:
                get_left(img)      #在终端输出左上角坐标，框大小和高
            else:
                auto_box(img,left1,w1,h1,name,save_path,left2,w2,h2)
                # auto_box(img,left1,w1,h1,'ir',save_path)
    else:
        path = 'E:/RTNIF/code/MSRS/'
        file_name = ['vis','ir']
        for name in file_name:
            img = cv2.imread(path + name + '/134.png')     #img_path
            bz = 0
            save_path = 'box/'  #保存路径
            if bz == 1:
                get_left(img)      #在终端输出左上角坐标，框大小和高
            else:
                auto_box(img,left1,w1,h1,name,save_path,left2,w2,h2)



