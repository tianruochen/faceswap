# -*- coding: utf-8 -*-
import numpy as np
import cv2
import dlib
import json
import ujson
import time
import base64
import requests


class faceswap:
    def __init__(self,DLIB_MODEL_PATH = './models/shape_predictor_68_face_landmarks.dat'):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
        self.bottom_info_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 26, 27, 28, 29, 30, 32, 33, 35, 36, 38, 40, 42]
        self.resize_size = 180
        ######################
        #初始化图像的脸部信息
        self.bottom_info_id = self.init_bottom_info()
        ######################


    def change_bottom(self,bottom_image):
        bottom_url = {1: 'http://tbfile.izuiyou.com/img/view/id/1036717357', 2: 'http://tbfile.izuiyou.com/img/view/id/1036717377',
         3: 'http://tbfile.izuiyou.com/img/view/id/1036717405', 4: 'http://tbfile.izuiyou.com/img/view/id/1036717426',
         5: 'http://tbfile.izuiyou.com/img/view/id/1036717439', 6: 'http://tbfile.izuiyou.com/img/view/id/1036717450',
         7: 'http://tbfile.izuiyou.com/img/view/id/1036717471', 8: 'http://tbfile.izuiyou.com/img/view/id/1036717489',
         9: 'http://tbfile.izuiyou.com/img/view/id/1036717514', 12: 'http://tbfile.izuiyou.com/img/view/id/1036717534',
         13: 'http://tbfile.izuiyou.com/img/view/id/1036717560', 14: 'http://tbfile.izuiyou.com/img/view/id/1036717574',
         15: 'http://tbfile.izuiyou.com/img/view/id/1036717596', 16: 'http://tbfile.izuiyou.com/img/view/id/1036717610',
         17: 'http://tbfile.izuiyou.com/img/view/id/1036717625', 18: 'http://tbfile.izuiyou.com/img/view/id/1036717662',
         19: 'http://tbfile.izuiyou.com/img/view/id/1036717688', 21: 'http://tbfile.izuiyou.com/img/view/id/1036717714',
         22: 'http://tbfile.izuiyou.com/img/view/id/1036717728', 23: 'http://tbfile.izuiyou.com/img/view/id/1036717750',
         26: 'http://tbfile.izuiyou.com/img/view/id/1036717767', 27: 'http://tbfile.izuiyou.com/img/view/id/1036717792',
         28: 'http://tbfile.izuiyou.com/img/view/id/1036717836', 29: 'http://tbfile.izuiyou.com/img/view/id/1036717904',
         30: 'http://tbfile.izuiyou.com/img/view/id/1036717939', 32: 'http://tbfile.izuiyou.com/img/view/id/1036717953',
         33: 'http://tbfile.izuiyou.com/img/view/id/1036717965', 35: 'http://tbfile.izuiyou.com/img/view/id/1036717973',
         36: 'http://tbfile.izuiyou.com/img/view/id/1036717987', 38: 'http://tbfile.izuiyou.com/img/view/id/1036718011',
         40: 'http://tbfile.izuiyou.com/img/view/id/1036718029', 42: 'http://tbfile.izuiyou.com/img/view/id/1036718037'}

        return bottom_url

    def init_bottom_info(self):
        l = [[1,1],[2,2],[3,1],[4,2],[5,1],[6,2],[7,2],[8,2],[9,1],[12,2],[13,2],[14,2],[15,1],[16,2],[17,1],[18,2],[19,2]
             ,[21,2],[22,2],[23,2],[26,2],[27,2],[28,1],[29,2],[30,2],[32,1],[33,2],[35,2],[36,2],[38,1],[40,2],[42,2]]
        bottom_img_info = {}
        for i in l:
            [bottom_id,_] = i
            # BOTTOM_IMAGE = '/Users/rhwang/Desktop/汉服活动底图的水印版/' + str(bottom_id) + '-水印.jpg'
            BOTTOM_IMAGE = './bottom_info/' + str(bottom_id) + '/' + str(bottom_id) + '.jpg'
            face_mask = cv2.imread('./bottom_info/' + str(bottom_id) + '/' + str(bottom_id) + '_.jpg')
            file = open('./bottom_info/' + str(bottom_id) + '/' + str(bottom_id) + '.json', 'r')
            [landmarks_bottom_list, r, correct_color] = json.load(file)
            bottom_img = self.imread(BOTTOM_IMAGE)  # 读取butom图像
            h, w, _ = bottom_img.shape
            w = (1600 * w) / (h * 1.0)
            bottom_img = cv2.resize(bottom_img, (int(w), int(1600)), interpolation=cv2.INTER_AREA)
            bottom_img_info[bottom_id] = [bottom_img,landmarks_bottom_list,face_mask,r, correct_color]
        return bottom_img_info
        ####################


    def image_url(self,img):
        url = 'http://opmedia.srv.in.ixiaochuan.cn/op/save_image'
        img_encode = cv2.imencode('.jpg', img)[1]
        file_64 = base64.b64encode(img_encode)
        ret_temp = requests.post(url, data=ujson.dumps({'file_data': file_64, 'buss_type': "zuiyou_img",
                                                        'internal_token': '5b8f785cd50a7d40714ff76d01699688'}
                                                       ))
        content = json.loads(ret_temp.content)
        if content['ret'] != 1:
            return 5
        image_id = content['data']['id']
        return 'http://tbfile.izuiyou.com/img/view/id/' + str(image_id)


    def get_landmarks(self,img):#获取人脸关键点68个
        landmarks = self.face_detector(img, 1)  # 获取脸部的框框
        if len(landmarks) == 0:
            print('there is no face')
            return 0
        elif len(landmarks) > 1:
            print('there are too many faces')
            return 1
        return np.matrix([[i.x, i.y] for i in self.shape_predictor(img, landmarks[0]).parts()])  # 获取68个关键点


    def imread(self,filename):#读取图像
        return cv2.imread(filename)


    def transformation_from_points(self,points1, points2):
        # 图片人脸对齐函数映射计算，输入两个关键点坐标矩阵返回一个对齐关系,后续可以改为透视变换
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        c1 = c1.reshape((1,2))
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2
        U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))
        R = (U * Vt).T
        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


    def warp_im(self,im, M, dshape):
        #f仿射变换
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output_im


    def correct_color(self,img1, img2, landmark,r):
        ##########################
        blur_amount = 0.4 * np.linalg.norm(
            np.mean(landmark[36:42], axis=0)
            - np.mean(landmark[42:48], axis=0)
        )
        blur_amount = int(blur_amount)

        if blur_amount % 2 == 0:
            blur_amount += 1
        ##########################

        ##########################
        #取出一人脸部分进行操作，这是为了加速
        h_plus1 = 10
        img1_ = img1[r[1]-h_plus1 :r[1]+int(r[3])+h_plus1,r[0]-h_plus1:r[0]+int(r[2])+h_plus1,:]
        img2_ = img2[r[1] - h_plus1:r[1] + int(r[3]) + h_plus1, r[0] - h_plus1:r[0] + int(r[2]) + h_plus1, :]
        ##########################


        ##########################
        #先模糊,后改色,
        img1_blur = cv2.GaussianBlur(img1_, (blur_amount, blur_amount), 0)
        img2_blur = cv2.GaussianBlur(img2_, (blur_amount, blur_amount), 0)
        img4 = img2.copy()
        img3 = img2_.astype(np.float32) * (img1_blur.astype(np.float32) / img2_blur.astype(np.float32))
        img3 = np.clip(img3, 0, 255).astype(np.uint8)
        ##########################

        ##########################
        #对脸部进行改色
        img4[r[1] - h_plus1:r[1] + int(r[3]) + h_plus1, r[0] - h_plus1:r[0] + int(r[2]) + h_plus1, :] = img3
        ##########################
        return img4


    def lightface(self,image,img,image_color):
        #############
        bottom_light = np.mean(cv2.cvtColor(image_color, cv2.COLOR_BGR2HLS)[:, :, 1])
        # bottom_light_clip = np.min(cv2.cvtColor(image_color, cv2.COLOR_BGR2HLS)[:, :, 1])
        a = np.mean(cv2.cvtColor(img,cv2.COLOR_BGR2HLS)[:,:,1])
        if a > bottom_light or a >175:
            return image
        else:
            image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            image1 = image1.astype(np.int32)
            image1[:, :, 1] = image1[:, :, 1]*(bottom_light/(a*1.0)) #+ (178-int(1.2*a))
            image1[:, :, 1] = np.clip(image1[:, :, 1], 0, 230)
            image1 = image1.astype(np.uint8)
            image = cv2.cvtColor(image1, cv2.COLOR_HLS2BGR)
            # Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)).show()
            return image
        #############


    def seg_face(self,l, img_, l1, img1,r):
        def find_mask(img, l):
            h_plus = l[29][1] - l[28][1]
            l1 = []
            for i in range(26, 16, -1):
                l1.append([l[i][0], l[i][1] - h_plus])

            lsPointsChoose = l[:17] + [l1[0], l1[1], l1[2], l1[7], l1[8], l1[9]]
            mask = np.zeros(img.shape, np.uint8)
            pts = np.array(lsPointsChoose, np.int32)  # pts是多边形的顶点列表（顶点集）
            # print(pts.shape)
            pts = pts.reshape((-1, 1, 2))
            mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
            mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
            return mask2

        #################
        # 选定指定的区域来取颜色
        w1 = max(l[1][0], l[2][0], l[3][0])
        h1 = l[1][1]
        w2 = int((l[1][0] + l[29][0]) / 2.0)
        h2 = l[2][1]
        # 选定指定的区域来取颜色
        w1_ = max(l1[1][0], l1[2][0], l1[3][0])
        h1_ = l1[1][1]
        w2_ = int((l1[1][0] + l1[29][0]) / 2.0)
        h2_ = l1[2][1]
        image_color = img1[h1_:h2_, w1_:w2_, :]
        #################

        #################
        # 制作脸部的mask
        mask2_ = find_mask(img_, l)
        mask3 = find_mask(img1, l1)
        h_plus1 = 20
        img = img_[r[1] - h_plus1:r[1] + int(r[3]) + h_plus1, r[0] - h_plus1:r[0] + int(r[2]) + h_plus1, :]
        mask2 = mask2_[r[1] - h_plus1:r[1] + int(r[3]) + h_plus1, r[0] - h_plus1:r[0] + int(r[2]) + h_plus1, :]
        mask_erosion = mask2
        #################

        #################
        # 大脸没话,制作边缘部分
        image_ = img_[h1:h2, w1:w2, :]
        img = self.lightface(img,image_,image_color)
        seg1 = cv2.bitwise_and(mask2, img)  # 大脸
        b, g, k, _ = cv2.mean(image_)
        edge_zeros = np.zeros(img.shape, img.dtype)
        edge_zeros[:, :] = np.array([int(b), int(g), int(k)])
        edge1 = cv2.bitwise_and(255 - mask2, edge_zeros)  # 大脸外部
        all_image1 = seg1 + edge1
        #################

        #################
        # 脸部外层的磨皮操作
        edge2 = cv2.bitwise_and(255 - mask_erosion, all_image1)  # 小脸包含外部轮廓的
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        edge2 = cv2.morphologyEx(edge2, cv2.MORPH_CLOSE, kernel)  # 进行开操作去处黑边
        #################

        #################
        seg2 = cv2.bitwise_and(mask_erosion, all_image1)  # 小脸
        seg3 = seg2 + edge2
        face_mask = np.zeros(img_.shape, np.uint8)
        face_mask[int(r[1] - h_plus1):int(r[1] + int(r[3]) + h_plus1), int(r[0] - h_plus1):int(r[0] + int(r[2]) + h_plus1),:] = seg3
        return face_mask, mask3
        #################


    def merge_img(self,bottom_img, mask_img, r,face_mask):
        center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
        return cv2.seamlessClone(np.uint8(mask_img), bottom_img, face_mask, center, cv2.NORMAL_CLONE)


    def edege_correct(self,img, r):
        h_plus1 = 20
        face_box = img[r[1]-h_plus1 :r[1]+int(r[3])+h_plus1,r[0]-h_plus1:r[0]+int(r[2])+h_plus1,:]
        face_box = cv2.bilateralFilter(face_box, 10, 30, 5)  # 双边美颜
        img[r[1] - h_plus1:r[1] + int(r[3]) + h_plus1, r[0] - h_plus1:r[0] + int(r[2]) + h_plus1, :] = face_box
        return img


    def faceswap_online(self,bottom_id,mask_img):
        '''
        :param bottom_id:must in self.bottom_info_id
        :param mask_img:Input image format is BGR
        :return: swaped_image, The return image format is BGR.  code msg url
        code msg url
        0  成功 url
        100 用户上传图像没有脸 ''
        200 用户上传图像脸太多 ''
        300 用户上传脸不完整  ''
        400 用户上传图像格式错误 ''
        500 获取换完脸的URL错误 ''
        '''
        ####################
        #判断输入id和图像格式是否正确
        if bottom_id not in self.bottom_info_id:
            return 500,'wrong bottom id',''
        try:
            h, w, channel = mask_img.shape
            if channel != 3:
                return 400,'wrong image type',''
            w = (self.resize_size * w) / (h * 1.0)
            ratio = h / self.resize_size*1.0
            mask_img1 = cv2.resize(mask_img, (int(w), int(self.resize_size)), interpolation=cv2.INTER_AREA)
        except:
            return 400,'wrong image type',''
        ####################

        ####################
        #获取底图人脸信息和输入图像信息
        [bottom_img, landmarks_bottom_list, face_mask, r, correct_color] = self.bottom_info_id[bottom_id]
        landmarks_bottom = np.array(landmarks_bottom_list)
        landmarks_mask1 = self.get_landmarks(mask_img1)
        landmarks_mask = ratio * landmarks_mask1
        ####################


        ####################
        #对输入图像的人脸数量和人脸关键点数量进行判断
        if type(landmarks_mask1) == int:
            if landmarks_mask1 == 0:
                return 100,'no face',''
            else:
                return 200,'too many face',''
        elif len(landmarks_mask1) != 68:
            return 300,'imcomplete face',''
        ####################

        ####################
        #利用普氏分析对图像进行旋转
        M = self.transformation_from_points(landmarks_bottom, landmarks_mask)
        warped_img = self.warp_im(mask_img, M, bottom_img.shape)
        ####################

        ####################
        #图像进行了旋转，对应人脸关键点也发生了变换，这里就行转换
        concate = np.concatenate((landmarks_mask, np.ones((68, 1), dtype=landmarks_mask.dtype)), axis=1)
        M = np.linalg.inv(M)
        landmarks_warped = np.dot(concate, M[:2].T)
        landmarks_warped_list = []
        for point in landmarks_warped:
            [[w, h]] = np.squeeze(point).tolist()
            landmarks_warped_list.append([int(w), int(h)])
        ####################

        ####################
        #获取底图人脸边界mask,并对wraped图像人脸进行分割
        warped_img,mask3 = self.seg_face(landmarks_warped_list, warped_img, landmarks_bottom_list, bottom_img,r)
        ####################

        ####################
        if correct_color == 1:
            warped_img = self.correct_color(bottom_img, warped_img, landmarks_bottom, r)
        ####################

        ####################
        # 泊松融合
        merged_img = self.merge_img(bottom_img, warped_img, r,face_mask)
        ####################

        ####################
        #人脸部分进行双边美颜
        merged_img = self.edege_correct(merged_img, r)
        try:
            image_url = self.image_url(merged_img)
            if image_url == 5:
                return 600,'wrong url',''
        except:
            return 600,'wrong url',''

        ####################
        return 0,'swapped face',image_url


if __name__ == '__main__':
    faceswap = faceswap()
    bottom_id = 1
    time1 = time.time()
    MASK_IMAGE = cv2.imread('/Users/rhwang/Desktop/汉服活动/mask_face/face80.png')
    print(time.time()-time1)
    time1 = time.time()
    aa,dd,fff = faceswap.faceswap_online(bottom_id, MASK_IMAGE)
