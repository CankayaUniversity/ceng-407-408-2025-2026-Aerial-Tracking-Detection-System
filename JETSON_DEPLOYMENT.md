# Jetson Nano Deployment & Docker Guide

Bu doküman, Aerial Tracking Detection System projesinin **Edge Node (Uç Cihaz)** tarafının Jetson Nano üzerinde Docker kullanılarak sıfırdan nasıl ayağa kaldırılacağını açıklamaktadır.

---

## 1. Faz: Dosya Transferi (Ana Bilgisayar -> Jetson)

Modelinizi, test videonuzu ve projeye ait olan Edge script'lerini Jetson'a aktarmanız gerekmektedir. Windows PowerShell veya Mac Terminalinizden şu komutları sırasıyla çalıştırın:

```bash
# Modellerinizin tamamını gönderin (-r ile)
scp -r models/ jetson@<JETSON_IP_ADRESI>:/home/jetson/models/

# Test videonuzu gönderin
scp test_video.mp4 jetson@<JETSON_IP_ADRESI>:/home/jetson/

# Edge Node Script'ini gönderin
scp scripts/jetson_edge_node.py jetson@<JETSON_IP_ADRESI>:/home/jetson/

# Tracker ve Inference kütüphanelerini gönderin (-r parametresi ile)
scp -r core/ jetson@<JETSON_IP_ADRESI>:/home/jetson/core/
```

---

## 2. Faz: Docker Konteynerinin Kurulumu ve Başlatılması

Bağımlılık çakışmalarını önlemek için NVIDIA'nın optimize edilmiş Ultralytics konteynerini kullanıyoruz. Jetson terminaline SSH ile bağlandıktan sonra şu komutu çalıştırın:

```bash
# --runtime nvidia: Konteynerin GPU'yu kullanabilmesini sağlar
# -v /home/jetson:/home/jetson : Dosya senkronizasyonu yapar (Jetson'a attığımız dosyalar anında içeride görünür)
# --network host: Jetson'un ağ üzerinden (UDP Broadcast) Ana Bilgisayarı bulabilmesi için ağ katmanını paylaşır
sudo docker run -it --network host --runtime nvidia -v /home/jetson:/home/jetson ultralytics/ultralytics:latest-aarch64 bash
```
> **Not:** Konteynerin Ana Bilgisayarı (UDP 50050 portu üzerinden) otomatik bulabilmesi için `--network host` (veya `--ipc=host` ile birlikte) parametresi hayati önem taşır.

---

## 3. Faz: Konteyner İçi Bağımlılıkların Kurulması

Konteynerin içine girdikten sonra (Terminal `root@...` olduğunda) sistemde eksik olabilecek kütüphaneleri yükleyin:

```bash
apt update
apt install nano -y
pip3 install opencv-python
```

---

## 4. Faz: TensorRT Optimizasyonu (Model Export)

Jetson üzerinde donanım hızlandırmalı maksimum FPS almak için, kullandığınız `.pt` modellerini `end2end=False` parametresiyle `.engine` formatına çevirmemiz şarttır.

Konteyner içerisinde şu komutu (kullandığınız her model için) çalıştırın:

```bash
python3 -c "from ultralytics import YOLO; model = YOLO('/home/jetson/models/rgb_normal.pt'); model.export(format='engine', device=0, half=True, imgsz=640, end2end=False, workspace=1024)"
```
*(Bunu `rgb_highlight.pt`, `ir_normal.pt` ve `ir_highlight.pt` modelleriniz için tekrarlayıp hepsini `.engine` formatına çevirebilirsiniz. Optimizasyon yapmazsanız `.pt` halleri de çalışacaktır ancak FPS düşük olur.)*

---

## 5. Faz: Sistemi Başlatma ve Ağ Bağlantısı

Artık her şey hazır. Çizim işlemleri (HUD) ve arayüz yükü Ana Bilgisayarda olduğu için Jetson sadece "headless" olarak çalışacaktır.

### Adım 5.1: Ana Bilgisayarı Hazırlama
Ana bilgisayarınızda (Windows/Mac) uygulamanızı başlatın ve Jetson'ı beklemeye alın:
1. `python3 main.py`
2. Üst menüden **"+ Add Edge Channel"** butonuna tıklayın.
3. **"Listen for Connection"** butonuna basarak dinlemeyi başlatın.

### Adım 5.2: Jetson Edge Node'u Başlatma
Konteynerinizin içindeyken aşağıdaki komutla test videonuz üzerinden takibi başlatın.

**Video Dosyası ile RGB Modunda Test (Bant Genişliği Tasarruflu):**
```bash
python3 /home/jetson/jetson_edge_node.py \
  --mode RGB \
  --video /home/jetson/test_video.mp4 \
  --no-video
```
*(Eğer `--base-model` ve `--motion-model` belirtmezseniz sistem otomatik olarak `models/rgb_normal.pt` (veya engine) yollarını kullanır.)*

**Kızılötesi (IR) Kamerası ile Canlı Kullanım:**
```bash
python3 /home/jetson/jetson_edge_node.py \
  --mode IR \
  --video 0
```
> Eğer modelleri engine formatına çevirdiyseniz komutun sonuna şunu ekleyin: `--base-model models/ir_normal.engine --motion-model models/ir_highlight.engine`

Kod başlatıldığı an `[*] Listening for Main Hub broadcast...` yazısı belirecek ve birkaç saniye içinde Ana bilgisayarınızı otomatik olarak bulup (`[+] Connected to Main Hub.`) işlemlere başlayacaktır.
