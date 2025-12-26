
if __name__ == '__main__':
    from ultralytics import YOLO

    # Load a model
    # Using a pretrained model like yolo11n-seg.pt is recommended for faster convergence
    model = YOLO("yolo11n-seg.yaml")

    # Train the model on the Crack Segmentation dataset
    # Ensure 'crack-seg.yaml' is accessible or provide the full path
    results = model.train(data="./datasets/crack-seg/data.yaml",             epochs=100,
            imgsz=640,
            batch=8,
            workers=0,
            lr0=0.001,
            amp=False)

    # After training, the model can be used for prediction or exported
    # results = model.predict(source='path/to/your/images')