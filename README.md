# 🏥 Wound Infection Detection

**كشف علامات العدوى في الجروح الجراحية باستخدام Deep Learning**

## ⚡ البيئة الموصى بها

**⚠️ مهم:** هذا المشروع يستخدم بيئة `.venv_cuda` مع:
- **Python 3.12.10**
- **PyTorch 2.5.1+cu121** (مع دعم CUDA)
- **CUDA 12.1**

البيئة جاهزة للاستخدام مع GPU (NVIDIA GeForce RTX 4060 أو أفضل).

## ⭐ مشروع منظم مع سكريبتات Python و Jupyter Notebooks

**`notebooks/train_model.py`** - سكريبت تدريب موحد شامل  
**`notebooks/training_pipeline.ipynb`** - Notebook للتدريب والتحليل

---

## 📁 هيكل المشروع

```
master_pro/
├── data/                          # البيانات (241 task)
│   ├── task_0/ ... task_240/     # البيانات الأصلية
│   ├── project.json
│   ├── annotations.json           # جميع البيانات (COCO format)
│   ├── splits/                    # تقسيمات البيانات
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── augmented/                 # البيانات المعززة (اختياري)
│       ├── annotations_augmented.json
│       └── images/
│
├── notebooks/
│   ├── train_model.py             # ⭐⭐ سكريبت التدريب الموحد (يدمج جميع وظائف التدريب)
│   ├── training_pipeline.ipynb    # Notebook للتدريب والتحليل
│   ├── pipeline_utils.py          # دوال معالجة البيانات
│   └── INFERENCE_GUIDE.md         # دليل الاستدلال والتحليل
│
├── scripts/                        # سكريبتات مساعدة
│   ├── apply_augmentation_only.py # تطبيق augmentation على البيانات
│   └── augmentation_strategy.py   # استراتيجية augmentation
│
├── docs/                           # التوثيق
│   └── DATA_AUGMENTATION_GUIDE.md # دليل augmentation
│
├── checkpoints/                    # النماذج المحفوظة
│   ├── best.pt                     # أفضل نموذج
│   └── last.pt                     # آخر checkpoint
├── checkpoints_medical_aug/        # نماذج مع augmentation
├── checkpoints_advanced/           # نماذج متقدمة
│
├── results/                        # النتائج (بعد Part 8)
│   └── *_result.json
│
├── requirements.txt                # المكتبات
├── setup_environment.bat           # إنشاء البيئة (Windows)
├── setup_environment.sh            # إنشاء البيئة (Linux/Mac)
├── run_jupyter.bat                 # تشغيل Jupyter (Windows)
├── run_jupyter.sh                  # تشغيل Jupyter (Linux/Mac)
└── README.md                       # هذا الملف
```

---

## 🚀 البدء السريع

### 1. التثبيت

#### 🐍 الطريقة الموصى بها: بيئة Python مع دعم CUDA

**⚠️ مهم:** هذا المشروع يستخدم بيئة `.venv_cuda` مع Python 3.12 و PyTorch مع دعم CUDA.

**Windows:**
```powershell
# إنشاء البيئة مع Python 3.12 (إذا لم تكن موجودة)
py -3.12 -m venv .venv_cuda

# تفعيل البيئة
.venv_cuda\Scripts\Activate.ps1

# تثبيت PyTorch مع CUDA
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# تثبيت باقي المكتبات
python -m pip install -r requirements.txt

# إعداد Jupyter Kernel
python -m ipykernel install --user --name=venv_cuda --display-name="Python 3.12 (CUDA)"
```

**Linux/Mac:**
```bash
# إنشاء البيئة مع Python 3.12
python3.12 -m venv .venv_cuda

# تفعيل البيئة
source .venv_cuda/bin/activate

# تثبيت PyTorch مع CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# تثبيت باقي المكتبات
pip install -r requirements.txt

# إعداد Jupyter Kernel
python -m ipykernel install --user --name=venv_cuda --display-name="Python 3.12 (CUDA)"
```

> **💡 الأفضل:** استخدم بيئة Python منفصلة لكل مشروع مع دعم CUDA

#### 📝 التحقق من CUDA

بعد التثبيت، تحقق من أن CUDA يعمل:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

**النتيجة المتوقعة:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

#### 📝 الطريقة اليدوية (Anaconda)

إذا كنت تستخدم Anaconda:
```bash
# 1. PyTorch (مع CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 2. المكتبات الأخرى
conda install opencv numpy pandas matplotlib seaborn -y
pip install -r requirements.txt
```

> **💡 نصيحة:** الأفضل استخدام بيئة `.venv_cuda` مع Python 3.12

### 2. طريقة الاستخدام

#### الطريقة 1: سكريبت Python (موصى به) 🚀

**التدريب المباشر:**
```bash
# من مجلد notebooks
cd notebooks
python train_model.py
```

**أو من جذر المشروع:**
```bash
python notebooks/train_model.py
```

#### الطريقة 2: Jupyter Notebook

**الطريقة الموصى بها:**
```powershell
# تفعيل البيئة
.venv_cuda\Scripts\Activate.ps1

# تشغيل Jupyter
jupyter notebook notebooks/training_pipeline.ipynb
```

**⚠️ مهم:** في Jupyter Notebook:
1. افتح `training_pipeline.ipynb`
2. اختر **Kernel → Change Kernel → Python 3.12 (CUDA)**
3. شغّل الخلايا - سيتم استخدام GPU تلقائياً

**أو يدوياً:**
```bash
# تفعيل البيئة أولاً
# Windows: .venv_cuda\Scripts\activate
# Linux/Mac: source .venv_cuda/bin/activate

jupyter notebook notebooks/training_pipeline.ipynb
```

### 3. شغّل الخلايا بالترتيب (في Notebook)

1. ✅ **Setup**: Import + Config
2. ⭐ **Data Loading**: تحميل البيانات
3. ✅ **Model Building**: بناء النموذج
4. ⭐⭐ **Training**: التدريب (4-6 ساعات)
5. ✅ **Evaluation**: التقييم
6. ⭐ **Inference**: التنبؤ والتحليل

---

## 📝 محتويات المشروع

### `train_model.py` - سكريبت التدريب الموحد

هذا الملف يدمج جميع وظائف التدريب في مكان واحد:

**وظائف بناء النموذج:**
- `build_model()` - بناء نموذج Mask R-CNN

**وظائف التدريب:**
- `train_one_epoch()` - تدريب epoch واحد
- `validate_one_epoch()` - التحقق من epoch واحد
- `main()` - دالة التدريب الرئيسية الكاملة

**وظائف التقييم:**
- `evaluate_metrics()` - تقييم المقاييس (COCO metrics)

**وظائف Checkpoints:**
- `save_checkpoint()` - حفظ checkpoint
- `load_checkpoint()` - تحميل checkpoint

**وظائف Inference:**
- `run_inference()` - تشغيل inference على صورة واحدة
- `run_wound_inference()` - inference خاص بحساب مساحة الجرح والعدوى

**وظائف التقارير:**
- `generate_report()` - توليد تقرير Markdown شامل

### `training_pipeline.ipynb` - Notebook للتدريب

**Setup & Configuration:**
- Import libraries
- CONFIG dictionary - عدّل الإعدادات هنا

**Data Loading:**
- تحميل البيانات من `data/splits/` أو `data/augmented/`
- دعم البيانات المعززة

**Model Building:**
- بناء النموذج باستخدام `train_model.build_model()`
- إعداد Optimizer & Scheduler

**Training:**
- حلقة التدريب الكاملة
- حفظ checkpoints تلقائياً

**Evaluation & Inference:**
- تقييم النموذج
- تشغيل inference على صور جديدة
- حساب مساحة الجرح وكشف العدوى

---

## ⚙️ التخصيص

### في `train_model.py`:

عدّل `CONFIG` في الملف:

```python
CONFIG = {
    # Data paths
    "data_root": "../data",
    "ann_file_train": "../data/splits/train.json",
    "ann_file_val": "../data/splits/val.json",
    
    # Training settings (uses GPU/CUDA when available)
    "device_prefer_cuda": True,
    "output_dir": "../checkpoints_medical_aug",
    "seed": 42,
    "batch_size": 4,
    "epochs": 50,
    "lr": 0.005,
    "image_size": (512, 512),
    
    # Medical Augmentation
    "use_medical_augmentation": True,
    "preserve_marker": True,
    "intensity": "moderate"  # "light", "moderate", "aggressive"
}
```

### في `training_pipeline.ipynb`:

عدّل `CONFIG` في الخلية الأولى:

```python
CONFIG = {
    "epochs": 50,
    "batch_size": 4,
    "lr": 0.005,
    "image_size": (512, 512),
    "use_medical_augmentation": False,  # True للـ augmentation أثناء التدريب
}
```

---

## 📊 المخرجات

### بعد تحضير البيانات:
- `data/annotations.json` - كل البيانات (COCO format)
- `data/splits/train.json` - بيانات التدريب
- `data/splits/val.json` - بيانات التحقق
- `data/splits/test.json` - بيانات الاختبار
- `data/augmented/` - البيانات المعززة (اختياري)

### بعد التدريب:
- `checkpoints_medical_aug/best.pt` - أفضل نموذج
- `checkpoints_medical_aug/last.pt` - آخر checkpoint
- `checkpoints_medical_aug/training_results.json` - نتائج التدريب
- `checkpoints_medical_aug/training_report.md` - تقرير شامل

### بعد Inference:
```json
{
  "wound_area_cm2": 25.3,
  "has_infection": true,
  "infection_confidence": 0.87,
  "findings": {
    "edema": true,
    "hyperemia": true,
    "necrosis": false,
    "granulation": true,
    "fibrin": true
  }
}
```

---

## 🎯 ما يكتشفه النظام

### العلامات الـ 16:

1. **AllWound** - كامل الجرح
2. **Fibrin** - الفيبرين
3. **SutureZone** - منطقة الخياطة
4. **EdemaZone** - الوذمة (علامة عدوى) ⚠️
5. **HyperemiaZone** - الاحتقان (علامة عدوى) ⚠️
6. **NecrosisZone** - النخر (علامة عدوى) ⚠️
7. **GranulationZone** - التحبب
8. **SizeMarker** - علامة القياس (3×3 سم)
9. وأكثر...

---

## 💡 نصائح

### إذا واجهت CUDA Out of Memory:
```python
# في Part 2، عدّل CONFIG:
CONFIG['batch_size'] = 1
CONFIG['image_size'] = [800, 800]
```

### للتدريب السريع:
```python
CONFIG['epochs'] = 10  # بدلاً من 50
```

### لمراقبة التدريب:
راقب الـ output في Notebook - سترى الـ loss ينخفض!

---

## 📈 النتائج المتوقعة

مع GPU (RTX 4060 أو أفضل):
- ⏱️ **التدريب**: 4-6 ساعات (50 epochs) على GPU
- ⏱️ **التدريب على CPU**: 20-30 ساعة (50 epochs) - **غير موصى به**
- 🎯 **mAP**: ~70-80%
- 🔍 **Infection Detection**: ~85%

**⚠️ مهم:** استخدم البيئة `.venv_cuda` للاستفادة من GPU وتقليل وقت التدريب بشكل كبير!

---

## 🆘 استكشاف الأخطاء

### ❌ CUDA غير متاح / PyTorch CPU-only

**المشكلة:** PyTorch مثبت بدون دعم CUDA

**الحل:**
1. تأكد من استخدام البيئة `.venv_cuda` (Python 3.12)
2. أعد تثبيت PyTorch مع CUDA:
   ```powershell
   .venv_cuda\Scripts\Activate.ps1
   pip uninstall torch torchvision -y
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. تحقق من CUDA:
   ```python
   import torch
   print(torch.cuda.is_available())  # يجب أن يطبع True
   ```

### ❌ ERROR: Unknown compiler / Preparing metadata failed

**المشكلة:** numpy يحاول البناء من المصدر (يتطلب Visual Studio)

**الحل:**
1. استخدم البيئة `.venv_cuda` (Python 3.12) - تحتوي على wheels جاهزة
2. أو شغّل: `pip install --only-binary :all: numpy scipy`

### ❌ ERROR: Could not install packages - WinError 32

**المشكلة:** pip لا يمكنه الوصول للملفات (مستخدمة من قبل عملية أخرى)

**الحل:**
1. **أغلق Jupyter Notebook** إذا كان مفتوحاً
2. **أغلق جميع نوافذ Terminal**
3. أعد المحاولة بعد إغلاق جميع العمليات
4. أو استخدم: `taskkill /F /IM python.exe` ثم أعد المحاولة

### ❌ ValueError: numpy.dtype size changed

**المشكلة:** تعارض بين numpy و scipy

**الحل:**
1. شغّل **Part 0.5** في Notebook (يصلح المشكلة تلقائياً)
2. أعد تشغيل Kernel: `Kernel → Restart`

### خطأ في تحميل البيانات؟
تأكد أن مجلد `data/` يحتوي على:
- `task_0/`, `task_1/`, ... `task_240/`
- `project.json`

### الـ loss لا ينخفض؟
- قلل `learning_rate` إلى 0.0005
- زد `epochs` إلى 100
- تأكد من البيانات صحيحة

### النموذج بطيء جداً؟
- قلل `image_size`
- قلل `batch_size`
- استخدم GPU أقوى

---

## 📚 المراجع

- **Mask R-CNN**: Instance Segmentation
- **PyTorch**: Framework التدريب
- **COCO Format**: صيغة البيانات

---

## 👨‍💻 المطور

مشروع رسالة ماجستير - كشف العدوى في الجروح الجراحية

---

**ملاحظة**: هذا مشروع بحثي. لا تستخدمه للقرارات الطبية الحقيقية دون استشارة طبية!

---

## 🎉 خلاصة

```
1 Jupyter Notebook = مشروع كامل
كل شيء منظم وواضح
جاهز للاستخدام فوراً
```

**ابدأ الآن!** 🚀

**الطريقة السريعة (سكريبت Python):**
```powershell
# تفعيل البيئة
.venv_cuda\Scripts\Activate.ps1

# تشغيل التدريب
cd notebooks
python train_model.py
```

**أو باستخدام Jupyter Notebook:**
```powershell
# تفعيل البيئة
.venv_cuda\Scripts\Activate.ps1

# تشغيل Jupyter
jupyter notebook notebooks/training_pipeline.ipynb

# ⚠️ مهم: اختر Kernel → Change Kernel → Python 3.12 (CUDA)
```

---

## 📚 الملفات الرئيسية

### `notebooks/train_model.py`
سكريبت Python موحد يحتوي على جميع وظائف التدريب والتقييم والاستدلال. يمكن تشغيله مباشرة أو استيراد دواله في notebooks أخرى.

**الاستخدام:**
```python
# تشغيل مباشر
python notebooks/train_model.py

# أو استيراد الدوال
from train_model import build_model, train_one_epoch, evaluate_metrics
```

### `notebooks/pipeline_utils.py`
دوال معالجة البيانات وإنشاء datasets:
- `create_dataset()` - إنشاء PyTorch Dataset
- `make_dataloaders()` - إنشاء DataLoaders
- `get_transforms()` - تحويلات الصور
- `WoundDataset` - Dataset class

### `scripts/apply_augmentation_only.py`
سكريبت لتطبيق augmentation على البيانات وحفظها:
```bash
cd scripts
python apply_augmentation_only.py
```

### `docs/DATA_AUGMENTATION_GUIDE.md`
دليل شامل لاستراتيجية augmentation الطبية.

### `notebooks/INFERENCE_GUIDE.md`
دليل استخدام وظائف inference والتحليل.
