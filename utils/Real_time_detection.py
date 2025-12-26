# import cv2
# from ultralytics import YOLO
#
# # 加载预训练的 YOLO 模型
# model = YOLO("runs/segment/train2/weights/best.pt")  # 替换为你的模型文件
#
# # 打开视频流（默认摄像头）
# #cap = cv2.VideoCapture(0)  # 0为默认摄像头，如果是视频文件，传入文件路径
# cap = cv2.VideoCapture('rode_te.mp4')
#
# # 检查视频流是否打开成功
# if not cap.isOpened():
#     print("Error: Could not open video stream.")
#     exit()
#
# # 设置输出视频的保存格式和属性
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码格式
# out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     # 模型推理（实时检测）
#     results = model(frame)
#
#     # 获取检测结果并渲染
#     frame = results[0].render()[0]  # 将检测框绘制在图像上
#
#     # 显示检测后的实时视频流
#     cv2.imshow("Real-time Object Detection", frame)
#
#     # 将处理后的帧写入输出视频
#     out.write(frame)
#
#     # 按 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放资源
# cap.release()
# out.release()  # 释放视频写入对象
# cv2.destroyAllWindows()


# import cv2
# from ultralytics import YOLO
# import numpy as np
#
# # 加载 YOLOv1-seg 模型
# model = YOLO("runs/segment/train2/weights/best.pt")  # 替换为你的 YOLOv1-seg 模型路径
#
# # 打开视频流（默认摄像头）
# # cap = cv2.VideoCapture(0)  # 0为默认摄像头，如果是视频文件，传入文件路径
# cap = cv2.VideoCapture('rode_te.mp4')
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 使用模型进行检测
#     #results = model(frame)  # 通过模型进行推断
#     results = model.predict(frame)
#     # 遍历每个检测结果
#     for result in results:
#         # 检查是否存在掩码
#         if result.masks is not None:
#             masks = result.masks.xy  # 获取实例的掩码坐标
#             classes = result.names  # 获取类别名称
#             confidences = result.boxes.conf  # 获取置信度
#             boxes = result.boxes.xywh  # 获取边界框坐标
#
#
#             for mask, cls, conf, box in zip(masks, classes, confidences, boxes):
#                 if conf > 0.5:  # 设置置信度阈值
#                     # 在图像上绘制掩码
#                     # 绘制边界框（如果需要显示框）
#                     x1, y1, w, h = map(int, box)
#                     cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
#                     color = (0, 255, 0) if cls == 'crack' else (0, 0, 255)  # 根据类别选择颜色
#                     for point in mask:
#                         point = np.array(point, dtype=np.int32).reshape((-1, 1, 2))  # 转换为 (N, 1, 2) 格式
#                         cv2.fillPoly(frame, [point], color)  # 使用掩码填充区域
#                         # 可以用 polylines 绘制多边形边界
#                         cv2.polylines(frame, [point], isClosed=True, color=color, thickness=2)
#         else:
#             print("No masks detected for this result")
#
#     # 显示检测结果
#     cv2.imshow("YOLOv8 Segmentation", frame)
#
#     # 按 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放视频流和窗口
# cap.release()
# cv2.destroyAllWindows()


import cv2
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("runs/segment/train2/weights/best.pt")  # 替换为你的 YOLOv8-seg 模型路径

# 打开视频流（默认摄像头）
cap = cv2.VideoCapture('end1mp4.mp4')  # 0为默认摄像头，如果是视频文件，传入文件路径

# 设置输出视频的保存格式和属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码格式
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 model.predict() 进行推断并获得结果
    results = model.predict(frame)  # 进行预测
    # 获取结果中的每个检测框的置信度、类别和坐标
    for result in results:
        boxes = result.boxes.xywh  # 获取边界框坐标
        confidences = result.boxes.conf  # 获取置信度
        classes = result.names  # 获取类别名称

        for box, conf, cls in zip(boxes, confidences, classes):
            if conf > 0.2:  # 只有当置信度大于设定阈值时才绘制
                frame = result.plot()
    # 将预测结果绘制到当前帧上
    # annotated_frame = results[0].plot()  # `results[0].plot()` 会直接给帧添加边界框和掩码

    # 将注释过的帧写入输出视频
    out.write(frame)

    # 显示当前帧
    cv2.imshow("YOLOv8 Segmentation", frame)
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频流和输出文件
cap.release()
out.release()
cv2.destroyAllWindows()




