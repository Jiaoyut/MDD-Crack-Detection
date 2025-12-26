import os

import cv2
import numpy as np

if __name__ == '__main__':
    from ultralytics import YOLO

    # Load a model
    # Using a pretrained model like yolo11n-seg.pt is recommended for faster convergence
    # model = YOLO("runs/segment/train11/weights/best.pt")
    # model = YOLO("runs/segment/train/weights/best.pt")
    model = YOLO("runs/segment/train2/weights/best.pt")

    # Train the model on the Crack Segmentation dataset
    # Ensure 'crack-seg.yaml' is accessible or provide the full path
    #results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640,device='cuda')

    # After training, the model can be used for prediction or exported
    results = model.predict(source='./testdata/',save=True)
    #results = model.predict(source='./datasets\crack-seg/test/images/images17_jpg.rf.8d3af2d4e5d030801821a980a6941e0a.jpg')
    # result = results[0]
    # result.show()

    # os.makedirs('masks', exist_ok=True)
    #
    # if result.masks is not None:
    #     masks = result.masks.data.cpu().numpy()  # (N, H, W)
    #     for i, mask in enumerate(masks):
    #         # 转换为 0-255 的 uint8 图像
    #         mask_img = (mask * 255).astype(np.uint8)
    #         # 保存为 PNG 文件
    #         cv2.imwrite(f"masks/mask_{i}.png", mask_img)
    #     print(f"已保存 {len(masks)} 个掩码到 masks/ 文件夹")
    # else:
    #     print("未检测到掩码")