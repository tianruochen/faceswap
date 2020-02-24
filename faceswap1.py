# -*- coding: utf-8 -*-
import numpy as np
import cv2
import dlib
from tqdm import tqdm
import time
from PIL import Image
from scipy.spatial import Delaunay




class faceswap:
    def __init__(self,DLIB_MODEL_PATH = '../faceswap1/shape_predictor_68_face_landmarks.dat'):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(DLIB_MODEL_PATH)

        ######################
        #用于人脸关键点
        self.JAW_POINTS = list(range(0, 27))
        self.JAW_END = 17
        self.FACE_END = 68
        self.OVERLAY_POINTS = [self.JAW_POINTS]  # LEFT_FACE, RIGHT_FACE,
        ######################


    def get_landmarks(self,img):#获取人脸关键点68个
        landmarks = self.face_detector(img, 1)  # 获取脸部的框框
        if len(landmarks) == 0:
            print('there is no face')
            return 0
        elif len(landmarks) > 1:
            print('there are too many faces')
            return 1
        return np.matrix([[i.x, i.y] for i in self.shape_predictor(img, landmarks[0]).parts()])  # 获取68个关键点


    def points_8(self,image, points):
        x = image.shape[1] - 1
        y = image.shape[0] - 1
        points = points.tolist()
        points.append([0, 0])
        points.append([x // 2, 0])
        points.append([x, 0])
        points.append([x, y // 2])
        points.append([x, y])
        points.append([x // 2, y])
        points.append([0, y])
        points.append([0, y // 2])
        return np.array(points)


    def imread(self,filename):#读取图像
        return cv2.imread(filename)


    def transformation_from_points(self,points1, points2):
        # 图片人脸对齐函数映射计算，输入两个关键点坐标矩阵返回一个对齐关系,后续可以改为透视变换
        # Procrustes 分析法
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
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


    def get_triangles(self,points):
        # 在特征点上使用 Delaunay 三角剖分
        return Delaunay(points).simplices


    def affine_transform(self,input_image, input_triangle, output_triangle, size):
        #
        warp_matrix = cv2.getAffineTransform(np.float32(input_triangle), np.float32(output_triangle))
        output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
        return output_image


    def morph_triangle(self,img1, img2, img, tri1, tri2, tri, alpha):
        # 计算三角形的边界框
        rect1 = cv2.boundingRect(np.float32([tri1]))
        rect2 = cv2.boundingRect(np.float32([tri2]))
        rect = cv2.boundingRect(np.float32([tri]))

        tri_rect1 = []
        tri_rect2 = []
        tri_rect_warped = []

        for i in range(0, 3):
            tri_rect_warped.append(
                ((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
            tri_rect1.append(
                ((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
            tri_rect2.append(
                ((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

        # 在边界框内进行仿射变换
        img1_rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
        img2_rect = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]

        size = (rect[2], rect[3])
        warped_img1 = self.affine_transform(img1_rect, tri_rect1, tri_rect_warped, size)
        warped_img2 = self.affine_transform(img2_rect, tri_rect2, tri_rect_warped, size)

        # 加权求和
        img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

        # 生成蒙版
        mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

        # 应用蒙版
        img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
            img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] * (1 - mask) + img_rect * mask


    def morph_face(self,bottom_img, mask_img, points1, points2, alpha=0.5):
        # 三角融合函数, 本质线性相加：M(x,y)=(1−α)I(x,y)+αJ(x,y)
        points1 = self.points_8(bottom_img, points1)
        points2 = self.points_8(mask_img, points2)
        morph_points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)
        bottom_img = np.float32(bottom_img)
        mask_img = np.float32(mask_img)
        img_morphed = np.zeros(bottom_img.shape, dtype=bottom_img.dtype)
        triangles = self.get_triangles(morph_points)
        for i in triangles:
            x = i[0]
            y = i[1]
            z = i[2]
            tri1 = [points1[x], points1[y], points1[z]]
            tri2 = [points2[x], points2[y], points2[z]]
            tri = [morph_points[x], morph_points[y], morph_points[z]]
            self.morph_triangle(bottom_img, mask_img, img_morphed, tri1, tri2, tri, alpha)
        return np.uint8(img_morphed)


    def affine_triangle(self,src, dst, t_src, t_dst):
        r1 = cv2.boundingRect(np.float32([t_src]))
        r2 = cv2.boundingRect(np.float32([t_dst]))

        t1_rect = []
        t2_rect = []
        t2_rect_int = []

        for i in range(0, 3):
            t1_rect.append((t_src[i][0] - r1[0], t_src[i][1] - r1[1]))
            t2_rect.append((t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]))
            t2_rect_int.append((t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]))

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

        img1_rect = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        size = (r2[2], r2[3])

        img2_rect = self.affine_transform(img1_rect, t1_rect, t2_rect, size)
        img2_rect = img2_rect * mask

        dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)
        dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


    def rect_contains(self,rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True


    def measure_triangle(self,image, points):
        rect = (0, 0, image.shape[1], image.shape[0])
        sub_div = cv2.Subdiv2D(rect)  # 画布
        points = points.tolist()
        for p in points:
            sub_div.insert((p[0], p[1]))  # 插入关键点
        triangle_list = sub_div.getTriangleList()  # 德劳力三角剖分
        triangle = []
        pt = []
        for t in triangle_list:
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            if self.rect_contains(rect, pt1) and self.rect_contains(rect, pt2) and self.rect_contains(rect, pt3):
                ind = []
                for j in range(0, 3):
                    for k in range(0, len(points)):
                        if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                            ind.append(k)
                if len(ind) == 3:
                    triangle.append((ind[0], ind[1], ind[2]))
            pt = []
        return triangle


    def tran_src(self,src_img, src_points, dst_points):
        jaw = self.JAW_END

        dst_list = self.points_8(src_img, dst_points)
        src_list = self.points_8(src_img, src_points)

        jaw_points = []

        for i in range(0, jaw):
            jaw_points.append(dst_list[i].tolist())
            jaw_points.append(src_list[i].tolist())

        warp_jaw = cv2.convexHull(np.array(jaw_points), returnPoints=False)
        warp_jaw = warp_jaw.tolist()

        for i in range(0, len(warp_jaw)):
            warp_jaw[i] = warp_jaw[i][0]

        warp_jaw.sort()

        if len(warp_jaw) <= jaw:
            dst_list = dst_list[jaw - len(warp_jaw):]
            src_list = src_list[jaw - len(warp_jaw):]
            for i in range(0, len(warp_jaw)):
                dst_list[i] = jaw_points[int(warp_jaw[i])]
                src_list[i] = jaw_points[int(warp_jaw[i])]
        else:
            for i in range(0, jaw):
                if len(warp_jaw) > jaw and warp_jaw[i] == 2 * i and warp_jaw[i + 1] == 2 * i + 1:
                    warp_jaw.remove(2 * i)

                dst_list[i] = jaw_points[int(warp_jaw[i])]

        dt = self.measure_triangle(src_img, dst_list)

        res_img = np.zeros(src_img.shape, dtype=src_img.dtype)

        for i in range(0, len(dt)):
            t_src = []
            t_dst = []

            for j in range(0, 3):
                t_src.append(src_list[dt[i][j]])
                t_dst.append(dst_list[dt[i][j]])
            self.affine_triangle(src_img, res_img, t_src, t_dst)
        return res_img


    def correct_color(self,img1, img2, landmark):
        blur_amount = 0.4 * np.linalg.norm(
            np.mean(landmark[36:42], axis=0)
            - np.mean(landmark[42:48], axis=0)
        )
        blur_amount = int(blur_amount)

        if blur_amount % 2 == 0:
            blur_amount += 1

        img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
        img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

        img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

        img3 = img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)
        return img3


    def seg_face(self,l, img, l1, img1):
        def find_mask(img, l):
            h_plus = l[29][1] - l[27][1]
            l1 = []
            for i in range(25, 17, -1):
                l1.append([l[i][0], l[i][1] - h_plus])
            lsPointsChoose = l[:17] + [[l[26][0]+h_plus*0.6, l[26][1]-h_plus]] + [l1[0], l1[1], l1[6],l1[7]] + [[l[17][0]-h_plus*0.6, l[17][1]-h_plus]]

            # l1 = []
            # for i in range(26, 16, -1):
            #     l1.append([l[i][0], l[i][1] - h_plus])
            #
            # lsPointsChoose = l[:17] + [l1[0], l1[1], l1[2], l1[7], l1[8], l1[9]]
            mask = np.zeros(img.shape, np.uint8)
            pts = np.array(lsPointsChoose, np.int32)  # pts是多边形的顶点列表（顶点集）
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
        #################

        #################
        # 制作脸部的mask
        mask2 = find_mask(img, l)
        mask3 = find_mask(img1, l1)
        #################

        #################
        # 图像的腐蚀，来留出边界来模糊
        erosion_len = l[34][0] - l[32][0]
        kernel = np.ones((int(erosion_len / 0.65), int(erosion_len / 0.5)), np.uint8)
        mask_erosion1 = cv2.erode(mask2, kernel)
        kernel = np.ones((int(erosion_len / 0.8), int(erosion_len / 0.7)), np.uint8)
        mask_erosion = cv2.erode(mask2, kernel)
        kernel = np.ones((int(erosion_len / 0.7), int(erosion_len / 0.6)), np.uint8)
        mask_erosion2 = cv2.erode(mask2, kernel)
        #################

        #################
        # 大脸没话,制作边缘部分
        seg1 = cv2.bitwise_and(mask2, img)  # 大脸
        b, g, r, _ = cv2.mean(seg1[h1:h2, w1:w2, :])
        edge_zeros = np.zeros(img.shape, img.dtype)
        edge_zeros[:, :] = np.array([int(b), int(g), int(r)])
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
        return seg3, mask_erosion1, mask_erosion2, mask3
        #################


    def merge_img(self,bottom_img, mask_img, mask_matrix, mask_points, blur_detail_x=None, blur_detail_y=None,
                  mat_multiple=None):
        # opencv泊松融合函数
        face_mask = np.zeros(bottom_img.shape, dtype=bottom_img.dtype)

        for group in self.OVERLAY_POINTS:
            face_points = mask_matrix[group]
            h_plus = mask_matrix[29][1] - mask_matrix[27][1]
            for i in range(16, 27):
                face_points[i][1] -= h_plus
            cv2.fillConvexPoly(face_mask, cv2.convexHull(face_points), (255, 255, 255))  # 填充人脸多边形
            r = cv2.boundingRect(np.float32([face_points]))
            center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

        ##########################
        #制作bottom_image
        # face_bottom1 = cv2.bitwise_and(face_mask, bottom_img)
        # face_bottom1 = cv2.medianBlur(face_bottom1, 101)
        # kernel = np.ones((int(30), int(30)), np.uint8)
        # face_mask1 = cv2.erode(face_mask, kernel)
        # face_bottom1 = cv2.bitwise_and(face_mask1, face_bottom1)
        # bottom_img = face_bottom1 + cv2.bitwise_and(255 - face_mask1, bottom_img)
        # # face_bottom1 = cv2.GaussianBlur(face_bottom1, ksize=(201, 201), sigmaX=0, sigmaY=0)
        # Image.fromarray(cv2.cvtColor(bottom_img.astype(np.uint8), cv2.COLOR_BGR2RGB)).show()
        # bottom_image = cv2.bitwise_and(255 - mask4, img1) + edge1
        ##########################

        if mat_multiple:
            mat = cv2.getRotationMatrix2D(center, 0, mat_multiple)
            face_mask = cv2.warpAffine(face_mask, mat, (face_mask.shape[1], face_mask.shape[0]))

        if blur_detail_x and blur_detail_y:
            face_mask = cv2.blur(face_mask, (blur_detail_x, blur_detail_y), center)
        return cv2.seamlessClone(np.uint8(mask_img), bottom_img, face_mask, center, cv2.NORMAL_CLONE)


    def edege_correct(self,img, mask1, mask_ersion2, mask2):
        mask2 = cv2.bitwise_or(mask1, mask2)
        face = cv2.bitwise_and(mask2, img)  # bottom_face
        face_in = cv2.bitwise_and(mask_ersion2, face)  # merged_face
        face_edge = cv2.bitwise_and(255 - mask1, face)
        face_edge_blur = cv2.bilateralFilter(face_edge, 0, 100, 5)  # 双边美颜
        face_edge_blur = cv2.bitwise_and(255 - mask_ersion2, face_edge_blur)
        face_new = face_in + face_edge_blur


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask4 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)  # 进行开操作去处黑边

        face_box_mask = mask4 - mask2
        mask5 = cv2.bitwise_and(face_box_mask, img)
        face_box = face_new + mask5
        face_box = cv2.bilateralFilter(face_box, 10, 30, 5)  # 双边美颜
        face_edge = cv2.bitwise_and(255 - mask4, img)  # 大脸外部

        # face_box_new = cv2.bilateralFilter(face_box, 10, 30, 5)  # 双边美颜
        all = face_box + face_edge
        return all


    def faceswap(self,BOTTOM_IMAGE,MASK_IMAGE,alpha = 0.95):
        bottom_img = self.imread(BOTTOM_IMAGE)  # 读取butom图像
        mask_img = self.imread(MASK_IMAGE)  # 读取mask图像
        landmarks_bottom = self.get_landmarks(bottom_img) # 获取人脸关键点68个
        landmarks_mask = self.get_landmarks(mask_img) # 获取人脸关键点68个

        landmarks_bottom_list = []
        for point in landmarks_bottom:
            [[w, h]] = np.squeeze(point).tolist()
            landmarks_bottom_list.append([w, h])

        ####################
        #利用普氏分析对图像进行旋转
        M = self.transformation_from_points(landmarks_bottom, landmarks_mask)
        warped_img = self.warp_im(mask_img, M, bottom_img.shape)
        ####################

        ####################
        #将图像人脸部分进行分割，为后面融合效果打基础
        concate = np.concatenate((landmarks_mask, np.ones((68, 1), dtype=landmarks_mask.dtype)), axis=1)
        M = np.linalg.inv(M)
        landmarks_warped = np.dot(concate, M[:2].T)

        landmarks_warped_list = []
        for point in landmarks_warped:
            [[w, h]] = np.squeeze(point).tolist()
            landmarks_warped_list.append([int(w), int(h)])
        warped_img, mask_ersion, mask_ersion2, mask3 = self.seg_face(landmarks_warped_list, warped_img, landmarks_bottom_list, bottom_img)
        ####################

        ####################
        #对图像进行三角融合
        morph_img = self.morph_face(bottom_img, warped_img, landmarks_bottom, landmarks_warped, float(alpha))
        ####################

        ####################
        # 修正融合图颜色与底图一致
        morph_image_revise = self.correct_color(bottom_img, morph_img, landmarks_bottom)
        cv2.imwrite('./out.jpg', morph_image_revise)
        morph_image_revise_imread = cv2.imread('./out.jpg')
        ####################

        ####################
        # 泊松融合,这里运行两次是为了效果好
        merged_img = self.merge_img(bottom_img, morph_image_revise_imread, np.array(landmarks_bottom_list), landmarks_bottom,
                               blur_detail_x=15,
                               blur_detail_y=10, mat_multiple=0.85)
        merged_img = self.edege_correct(merged_img, mask_ersion, mask_ersion2, mask3)
        # merged_img = self.merge_img(bottom_img, merged_img, np.array(landmarks_bottom_list), landmarks_bottom, blur_detail_x=15,
        #                        blur_detail_y=10, mat_multiple=0.95)
        ###################
        return merged_img


    def faceswap_bottom(self,bottom_img,mask_img,landmarks_bottom,alpha = 0.95):

        h, w, _ = mask_img.shape
        w = (250 * w) / (h * 1.0)
        mask_img = cv2.resize(mask_img, (int(w), int(250)), interpolation=cv2.INTER_AREA)
        landmarks_mask = self.get_landmarks(mask_img)
        if type(landmarks_mask) == int or type(landmarks_mask) == float:
            return 0
        elif len(landmarks_mask) != 68:
            return 0
        landmarks_bottom_list = []
        for point in landmarks_bottom:
            [[w, h]] = np.squeeze(point).tolist()
            landmarks_bottom_list.append([w, h])

        ####################
        #利用普氏分析对图像进行旋转
        M = self.transformation_from_points(landmarks_bottom, landmarks_mask)
        warped_img = self.warp_im(mask_img, M, bottom_img.shape)
        ####################

        ####################
        #将图像人脸部分进行分割，为后面融合效果打基础
        # landmarks_warped = self.get_landmarks(warped_img)
        concate = np.concatenate((landmarks_mask, np.ones((68, 1), dtype=landmarks_mask.dtype)), axis=1)
        M = np.linalg.inv(M)
        landmarks_warped = np.dot(concate, M[:2].T)

        if type(landmarks_mask) == int:
            return 0
        elif len(landmarks_mask) != 68:
            return 0
        landmarks_warped_list = []
        try:
            for point in landmarks_warped:
                [[w, h]] = np.squeeze(point).tolist()
                landmarks_warped_list.append([int(w), int(h)])
        except:
            return 0
        warped_img, mask_ersion, mask_ersion2, mask3 = self.seg_face(landmarks_warped_list, warped_img, landmarks_bottom_list, bottom_img)
        ####################

        ####################
        #对图像进行三角融合
        # morph_img = self.morph_face(bottom_img, warped_img, landmarks_bottom, landmarks_warped, float(alpha))
        morph_img = alpha * warped_img + (1 - alpha) * bottom_img
        ####################

        ####################
        # 修正融合图颜色与底图一致
        morph_image_revise = self.correct_color(bottom_img, morph_img, landmarks_bottom)
        cv2.imwrite('./out.jpg', morph_image_revise)
        morph_image_revise_imread = cv2.imread('./out.jpg')
        ####################

        ####################
        # 泊松融合,这里运行两次是为了效果好
        merged_img = self.merge_img(bottom_img, morph_image_revise_imread, np.array(landmarks_bottom_list), np.array(landmarks_bottom_list),
                               blur_detail_x=15,
                               blur_detail_y=10, mat_multiple=0.85)

        # merged_img = self.edege_correct(merged_img, mask_ersion, mask_ersion2, mask3)
        # merged_img = self.merge_img(bottom_img, merged_img, np.array(landmarks_bottom_list), np.array(landmarks_bottom_list), blur_detail_x=15,
        #                        blur_detail_y=10, mat_multiple=0.95)
        ####################
        return merged_img


    def faceswap_fromvideo(self,BOTTOM_IMAGE,alpha = 0.999):
        bottom_img = self.imread(BOTTOM_IMAGE)  # 读取butom图像
        h,w,_ = bottom_img.shape
        w = (300*w)/(h*1.0)
        bottom_img = cv2.resize(bottom_img, (int(w),int(300)), interpolation=cv2.INTER_AREA)
        landmarks_bottom = self.get_landmarks(bottom_img)  # 获取人脸关键点68个

        if type(landmarks_bottom) == int:
            return
        cap = cv2.VideoCapture(0)
        while True:
            # 从摄像头读取图片
            sucess, mask_img = cap.read()
            w,h,_ = mask_img.shape

            mask_img = cv2.resize(mask_img, (int(h/2.0),int(w/2.0)), interpolation=cv2.INTER_AREA)
            # 显示摄像头，背景是灰度。
            time1 = time.time()
            faceswaped = self.faceswap_bottom(bottom_img,mask_img,landmarks_bottom,alpha)
            print(time.time()-time1)
            if type(faceswaped) == int:
                continue
            cv2.imshow('mask_image',mask_img)
            cv2.imshow("faceswaped_image", faceswaped)
            # 保持画面的持续。
            k = cv2.waitKey(5)
            if k == 27:
                # 通过esc键退出摄像
                cv2.destroyAllWindows()
                break
            elif k == ord("s"):
                # 通过s键保存图片，并退出。
                cv2.imwrite("faceswaped.jpg", faceswaped)
                cv2.destroyAllWindows()
                break
        # 关闭摄像头
        cap.release()


if __name__ == '__main__':
    faceswap = faceswap()
    BOTTOM_IMAGE = './images/face47.jpg'
    merged_img = faceswap.faceswap_fromvideo(BOTTOM_IMAGE)


    # faceswap = faceswap()
    # BOTTOM_IMAGE = './images/face18.jpeg'
    # MASK_IMAGE = './images/face21.png'
    #
    # merged_img = faceswap.faceswap(BOTTOM_IMAGE, MASK_IMAGE, alpha=0.99)
    # Image.fromarray(cv2.cvtColor(merged_img.astype(np.uint8), cv2.COLOR_BGR2RGB)).show()
    #






