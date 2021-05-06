from scipy.spatial import distance as dis
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import numpy
import time
import dlib
import cv2
import winsound
import math

# 世界坐标系(UVW)：填写3D参考点
object_pts = numpy.float32([[6.825897, 6.760612, 4.402142],  #33左眉左上角
                         [1.330353, 7.122144, 6.903745],  #29左眉右角
                         [-1.330353, 7.122144, 6.903745], #34右眉左角
                         [-6.825897, 6.760612, 4.402142], #38右眉右上角
                         [5.311432, 5.485328, 3.987654],  #13左眼左上角
                         [1.789930, 5.393625, 4.413414],  #17左眼右上角
                         [-1.789930, 5.393625, 4.413414], #25右眼左上角
                         [-5.311432, 5.485328, 3.987654], #21右眼右上角
                         [2.005628, 1.409845, 6.165652],  #55鼻子左上角
                         [-2.005628, 1.409845, 6.165652], #49鼻子右上角
                         [2.774015, -2.080775, 5.048531], #43嘴左上角
                         [-2.774015, -2.080775, 5.048531],#39嘴右上角
                         [0.000000, -3.116408, 6.097667], #45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])#6下巴角
# 相机坐标系(XYZ)：添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]# 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
# 像素坐标系(xy)：填写凸轮的本征和畸变系数
cam_matrix = numpy.array(K).reshape(3, 3).astype(numpy.float32)
dist_coeffs = numpy.array(D).reshape(5, 1).astype(numpy.float32)
# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = numpy.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

#计算眼睛的长宽比
def eyesRatio(eye):
    left = dis.euclidean(eye[1], eye[5])
    right = dis.euclidean(eye[2], eye[4])
    horizontal = dis.euclidean(eye[0], eye[3])
    return 2.0*horizontal/(left+right)

def mouth_aspect_ratio(mouth):# 嘴部
    A = numpy.linalg.norm(mouth[2] - mouth[9])
    B = numpy.linalg.norm(mouth[4] - mouth[7])
    C = numpy.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


def get_head_pose(shape):  # 头部姿态估计
    # （像素坐标集合）填写2D参考点
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    image_pts = numpy.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    return reprojectdst, euler_angle  # 投影误差，欧拉角

#创建一个解析对象，向该对象中添加关注的命令行参数和选项，然后解析
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0)
args = vars(ap.parse_args())

#眼睛长宽比的阈值，如果超过这个值就代表眼睛长/宽大于采集到的平均值，默认已经"闭眼"
eyesRatioLimit=0
#数据采集的计数，采集30次然后取平均值
collectCount=0
#用于数据采集的求和
collectSum=0
#是否开始检测
startCheck=False
#统计"闭眼"的次数
eyesCloseCount=0
# 初始化帧计数器和打哈欠总数
mCOUNTER = 0
mTOTAL = 0
# 初始化帧计数器和点头总数
hCOUNTER = 0
hTOTAL = 0

#初始化dlib
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("C:\\Users\\master\\Desktop\\shape_predictor_68_face_landmarks.dat")

#左右眼和面部
(left_Start,left_End)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_Start,right_End)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#开启视频线程，延迟2秒钟
vsThread=VideoStream(src=args["webcam"]).start()
time.sleep(2.0)

#循环检测
while True:
    #对每一帧进行处理，设置宽度并转化为灰度图
    frame = vsThread.read()
    frame = imutils.resize(frame, width=720)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 待会要写的字体
    font = cv2.FONT_HERSHEY_COMPLEX
    #检测灰度图中的脸
    faces = detector(img, 0)
    for k in faces:
        #确定面部区域的面部特征点，将特征点坐标转换为numpy数组
        shape = predictor(img, k)
        shape = face_utils.shape_to_np(shape)

        #左右眼
        leftEye = shape[left_Start:left_End]
        rightEye = shape[right_Start:right_End]
        leftEyesVal = eyesRatio(leftEye)
        rightEyesVal = eyesRatio(rightEye)
        #凸壳
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #取两只眼长宽比的的平均值作为每一帧的计算结果
        eyeRatioVal = (leftEyesVal + rightEyesVal) / 2.0

        # 嘴巴坐标
        mouth = shape[mStart:mEnd]
        # 打哈欠
        mar = mouth_aspect_ratio(mouth)

        # 进行画图操作，用矩形框标注人脸
        left = k.left()
        top = k.top()
        right = k.right()
        bottom = k.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
        #判断眼睛是否疲劳
        if collectCount<30:
            collectCount+=1
            collectSum+=eyeRatioVal
            cv2.putText(frame, "RUNNING", (300, 30),font, 0.6, (0, 0, 255), 2)
            startCheck=True
        else:
            if startCheck:
                eyesRatioLimit=collectSum/30
                #如果眼睛长宽比大于之前检测到的阈值，则计数，闭眼次数超过50次则提示疲劳
                if eyeRatioVal > eyesRatioLimit:
                    eyesCloseCount += 1
                    if eyesCloseCount >= 50:
                        cv2.putText(frame, "WARNING!!!", (580, 30),font, 0.8, (0, 0, 255), 2)
                        winsound.Beep(1000, 1000)  # 调用提示音，设置声音频率与持续时间
                else:
                    eyesCloseCount = 0
                #眼睛长宽比
                cv2.putText(frame, "EYES_RATIO: {:.2f}".format(eyeRatioVal), (20, 30), font, 0.6, (0, 0, 255), 2)
                #闭眼次数
                cv2.putText(frame,"EYES_COLSE: {}".format(eyesCloseCount), (300, 30), font,0.6,(0, 0, 255),2)
        # 判断是否打哈欠
        if mar > 0.5:  # 张嘴阈值0.5
            mCOUNTER += 1
        else:
            # 如果连续3次都大于阈值，则表示打了一次哈欠
            if mCOUNTER >= 25:  # 阈值：25
                mTOTAL += 1
            # 重置嘴帧计数器
            mCOUNTER = 0
        cv2.putText(frame, "MCOUNTER: {}".format(mCOUNTER), (20, 60), font, 0.7,(0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (250, 60), font, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "mTOTAL: {}".format(mTOTAL), (400, 60), font, 0.7,(255, 255, 0), 2)

        # 判断是否瞌睡点头
        reprojectdst, euler_angle = get_head_pose(shape)

        har = euler_angle[0, 0]
        if har > 15:  # 点头阈值15
            hCOUNTER += 1
        else:
            # 如果连续3次都大于阈值，则表示瞌睡点头一次
            if hCOUNTER >= 30:  # 阈值：30
                hTOTAL += 1
            # 重置点头帧计数器
            hCOUNTER = 0
        cv2.putText(frame, "HCOUNTER: {}".format(hCOUNTER), (20, 90), font, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "HAR: {:.2f}".format(har), (250, 90), font, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "hTOTAL: {}".format(hTOTAL), (400, 90), font, 0.7, (255, 255, 0), 2)

        # 确定疲劳判定:打哈欠15次，瞌睡点头15次
        if mTOTAL >= 15 or hTOTAL >= 15:
            cv2.putText(frame, "TIRED!!!", (100, 200),font, 1, (0, 0, 255), 3)

    # 显示人脸数
    cv2.putText(frame, "Faces: " + str(len(faces)), (20, 400), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "the space key: Quit", (20, 450), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Detection screen", frame)

    key = cv2.waitKey(1)
    #用空格键停止检测
    if key == ord(" "):
        break

cv2.destroyAllWindows()
vsThread.stop()