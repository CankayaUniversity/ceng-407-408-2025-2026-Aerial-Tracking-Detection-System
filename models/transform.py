import os
from ultralytics import YOLO

# Modellerin olduğu tam yol
model_dir = r"C:\Users\Kaan\Downloads\ceng-407-408-2025-2026-Aerial-Tracking-Detection-System (2)\ceng-407-408-2025-2026-Aerial-Tracking-Detection-System\models"

# Klasördeki tüm .pt dosyalarını listele
pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]

print(f"Bulunan model sayısı: {len(pt_files)}")

for file_name in pt_files:
    full_path = os.path.join(model_dir, file_name)
    print(f"\n--- Dönüştürülüyor: {file_name} ---")
    
    try:
        # Modeli yükle
        model = YOLO(full_path)
        
        # Jetson Nano (TensorRT 8.x) için en güvenli ayarlar:
        # imgsz: 640 (Performans dengesi)
        # opset: 12 (Nano'nun eski TensorRT'si için HAYATİ)
        # simplify: True (Gereksiz katmanları siler, hızı artırır)
        model.export(format="onnx", imgsz=640, opset=12, end2end=False, simplify=True)
        
        print(f"✅ Başarılı: {file_name} -> ONNX")
    except Exception as e:
        print(f"❌ Hata oluştu ({file_name}): {e}")

print("\nİşlem tamamlandı! ONNX dosyaları modeller klasöründe oluşturuldu.")