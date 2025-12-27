
from ultralytics import YOLO

# --------------------------------------------------------------------------------
# 3. EĞİTİM ANA DÖNGÜSÜ
# --------------------------------------------------------------------------------

if __name__ == '__main__':


    # --------------------------------------------------------------------------------
        # Incremental learning için bu satırları kullan
    # --------------------------------------------------------------------------------
    #motion_model_path = r'C:\Users\USER\Desktop\DRONE\runs\detect\3_channel14\weights\best.pt'
    #model_path = r'C:\Users\USER\Desktop\DRONE\runs\detect\without2\weights\best.pt'
    #model = YOLO(model_path)

    # --------------------------------------------------------------------------------
        # İlk training için
    # --------------------------------------------------------------------------------
    model = YOLO("yolo12s.pt")
    model.train(

        data='data.yaml',
        epochs=10,
        imgsz=640,
        batch=32,
        device=0,
        lr0=0.001,   # Her sessionda LR yi düşür

        optimizer='AdamW',
        name='without3',
        pretrained=True,
        val = True

    )

