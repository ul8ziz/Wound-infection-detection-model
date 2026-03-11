# 🐍 Installing libraries - Python 3.13

## ⚡ Direct commands (copy and paste)

```bash
# 1. Install setuptools and wheel (very important!)
python -m pip install --upgrade setuptools wheel

# 2. Install numpy and scipy (versions compatible with Python 3.13)
python -m pip install numpy>=1.26.0 scipy>=1.11.0

# 3. Install PyTorch
python -m pip install torch torchvision

# 4. Install Computer Vision
python -m pip install opencv-python Pillow albumentations

# 5. Install Data Processing
python -m pip install pandas

# 6. Install Visualization
python -m pip install matplotlib seaborn

# 7. Install Progress & Metrics
python -m pip install tqdm scikit-learn pycocotools

# 8. Install Config & Notebooks
python -m pip install pyyaml jupyter ipywidgets
```

---

## ✅ Verify installation

```bash
python -c "import torch; import cv2; import numpy as np; print(f'✓ NumPy: {np.__version__}'); print('✓ All packages installed!')"
```

---

## 🎯 Alternative: Use Python 3.12

If problems persist, use Python 3.12 (more stable):

1. Download Python 3.12 from [python.org](https://www.python.org/downloads/)
2. Create a new environment:
   ```bash
   py -3.12 -m venv venv
   venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

---

**Ready!** 🚀
