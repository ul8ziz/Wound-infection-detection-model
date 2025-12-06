# 🏥 Wound Infection Detection

**كشف علامات العدوى في الجروح الجراحية باستخدام Deep Learning**

## ⭐ مشروع كامل في ملف Jupyter Notebook واحد!

**`notebooks/complete_pipeline.ipynb`** - كل شيء من البداية للنهاية

---

## 📁 هيكل المشروع

```
master_pro/
├── data/                          # البيانات (241 task)
│   ├── task_0/ ... task_240/
│   ├── project.json
│   ├── annotations.json           # يُنشأ بعد Part 4
│   └── splits/                    # يُنشأ بعد Part 4
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── notebooks/
│   └── complete_pipeline.ipynb    # ⭐⭐ المشروع الكامل!
│
├── checkpoints/                    # النماذج (بعد Part 6)
│   └── best_model.pth
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

#### 🐍 الطريقة الموصى بها: بيئة Python جديدة

**Windows:**
```bash
# إنشاء البيئة وتثبيت المكتبات
setup_environment.bat

# تشغيل Jupyter
run_jupyter.bat
```

**Linux/Mac:**
```bash
# إنشاء البيئة وتثبيت المكتبات
chmod +x setup_environment.sh
./setup_environment.sh

# تشغيل Jupyter
chmod +x run_jupyter.sh
./run_jupyter.sh
```

> **💡 الأفضل:** استخدم بيئة Python منفصلة لكل مشروع

#### 📝 الطريقة اليدوية

إذا كنت تستخدم Anaconda:
```bash
# 1. PyTorch (مع CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 2. المكتبات الأخرى
conda install opencv numpy pandas matplotlib seaborn -y
pip install -r requirements.txt
```

> **💡 نصيحة:** الأفضل استخدام بيئة Python منفصلة (setup_environment.bat)

### 2. افتح Notebook

**إذا استخدمت البيئة الافتراضية:**
```bash
# Windows
run_jupyter.bat

# Linux/Mac
./run_jupyter.sh
```

**أو يدوياً:**
```bash
# تفعيل البيئة أولاً
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

jupyter notebook notebooks/complete_pipeline.ipynb
```

### 3. شغّل الخلايا بالترتيب

في Notebook:

1. ✅ **Part 1-3**: Setup (Import + Config + Functions)
2. ⭐ **Part 4**: تحضير البيانات (مرة واحدة فقط)
3. 📊 **Part 4.5**: تحليل البيانات (اختياري)
4. ✅ **Part 5**: إعداد النموذج
5. ⭐⭐ **Part 6**: التدريب (4-6 ساعات)
6. ✅ **Part 7**: Prediction Functions
7. ⭐ **Part 8**: التنبؤ (عدّل `image_path` أولاً)

---

## 📝 محتويات Notebook

### Part 1: Import Libraries
كل المكتبات المطلوبة

### Part 2: Configuration
`CONFIG` dictionary - عدّل الإعدادات هنا

### Part 3: Data Processing
- `convert_cvat_to_coco()` - تحويل CVAT → COCO
- `split_dataset()` - تقسيم البيانات
- `WoundDataset` - PyTorch Dataset

### Part 4: Run Data Preparation ⭐
شغّل مرة واحدة لتحضير البيانات

### Part 4.5: Data Analysis (اختياري)
إحصائيات سريعة عن البيانات

### Part 5: Model Building & Training
- `build_model()` - Mask R-CNN
- Datasets & DataLoaders
- Training functions
- Optimizer & Scheduler

### Part 6: Start Training ⭐⭐
حلقة التدريب الكاملة

### Part 7: Prediction Functions
- `calculate_wound_area()` - حساب المساحة
- `detect_infection()` - كشف العدوى
- `predict_image()` - التنبؤ
- `visualize_prediction()` - الرسم

### Part 8: Run Prediction ⭐
عدّل `image_path` ثم شغّل

---

## ⚙️ التخصيص

عدّل `CONFIG` في **Part 2**:

```python
CONFIG = {
    'epochs': 50,              # عدد الـ epochs
    'batch_size': 2,           # حجم الـ batch
    'learning_rate': 0.001,    # معدل التعلم
    'image_size': [1024, 1024], # حجم الصورة
    'device': 'cuda',          # أو 'cpu'
}
```

---

## 📊 المخرجات

### بعد Part 4:
- `data/annotations.json` - كل البيانات
- `data/splits/train.json` - بيانات التدريب
- `data/splits/val.json` - بيانات التحقق
- `data/splits/test.json` - بيانات الاختبار

### بعد Part 6:
- `checkpoints/best_model.pth` - أفضل نموذج
- `checkpoints/checkpoint_epoch_*.pth` - checkpoints دورية

### بعد Part 8:
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

مع GPU قوي (RTX 3090):
- ⏱️ **التدريب**: 4-6 ساعات (50 epochs)
- 🎯 **mAP**: ~70-80%
- 🔍 **Infection Detection**: ~85%

---

## 🆘 استكشاف الأخطاء

### ❌ ERROR: Unknown compiler / Preparing metadata failed

**المشكلة:** numpy يحاول البناء من المصدر (يتطلب Visual Studio)

**الحل:**
1. تم تحديث `requirements.txt` لاستخدام numpy 1.24.3 (wheel جاهز)
2. شغّل: `install_prebuilt.bat` (يستخدم wheels جاهزة فقط)
3. أو شغّل `setup_environment.bat` مرة أخرى

### ❌ ERROR: Could not install packages - WinError 32

**المشكلة:** pip لا يمكنه الوصول للملفات (مستخدمة من قبل عملية أخرى)

**الحل:**
1. **أغلق Jupyter Notebook** إذا كان مفتوحاً
2. **أغلق جميع نوافذ Terminal**
3. شغّل: `fix_pip_error.bat` (يغلق Python تلقائياً)
4. أو شغّل `setup_environment.bat` مرة أخرى

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

```bash
jupyter notebook notebooks/complete_pipeline.ipynb
```
