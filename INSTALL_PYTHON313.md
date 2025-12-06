# 🐍 تثبيت المكتبات - Python 3.13

## ⚡ الأوامر المباشرة (انسخ والصق)

```bash
# 1. تثبيت setuptools و wheel (مهم جداً!)
python -m pip install --upgrade setuptools wheel

# 2. تثبيت numpy و scipy بإصدارات تدعم Python 3.13
python -m pip install numpy>=1.26.0 scipy>=1.11.0

# 3. تثبيت PyTorch
python -m pip install torch torchvision

# 4. تثبيت Computer Vision
python -m pip install opencv-python Pillow albumentations

# 5. تثبيت Data Processing
python -m pip install pandas

# 6. تثبيت Visualization
python -m pip install matplotlib seaborn

# 7. تثبيت Progress & Metrics
python -m pip install tqdm scikit-learn pycocotools

# 8. تثبيت Config & Notebooks
python -m pip install pyyaml jupyter ipywidgets
```

---

## ✅ التحقق من التثبيت

```bash
python -c "import torch; import cv2; import numpy as np; print(f'✓ NumPy: {np.__version__}'); print('✓ All packages installed!')"
```

---

## 🎯 الخيار البديل: استخدام Python 3.12

إذا استمرت المشاكل، استخدم Python 3.12 (أكثر استقراراً):

1. حمّل Python 3.12 من [python.org](https://www.python.org/downloads/)
2. أنشئ بيئة جديدة:
   ```bash
   py -3.12 -m venv venv
   venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

---

**جاهز!** 🚀

