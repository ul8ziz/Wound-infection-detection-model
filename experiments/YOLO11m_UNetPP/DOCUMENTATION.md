# توثيق تجربة YOLO11m + U-Net++

هذا الملف يشرح **كل مرحلة تشغيل** و**كل عملية** في مجلد `experiments/YOLO11m_UNetPP`: من تحضير البيانات إلى التدريب والتقييم والاستدلال المدمج.

**الملفات الأساسية:**

| الملف | الدور |
|--------|--------|
| `config.yaml` | جميع المعاملات (بيانات، YOLO، U-Net++، الاستدلال المدمج) |
| `pipeline_utils.py` | تحويل COCO→YOLO، مجموعات بيانات U-Net++ و`WoundDataset`، تحويلات Albumentations |
| `train_model.py` | واجهة CLI لجميع المراحل، التدريب، التقييم، التقرير |
| `training_pipeline.ipynb` | نفس المنطق بشكل تفاعلي |

**البيانات:** مأخوذة من جذر المشروع، لا تُنسخ داخل مجلد التجربة. المسارات الافتراضية في `config.yaml` تشير إلى `data/wound_focus_clean` وملفات COCO *wound-only*.

---

## 1. مبدأ التجربة (معماريتان مستقلتان ثم دمج)

1. **YOLO11m-seg:** كشف الجرح على صورة كاميرة + قناع تجزئة أولي (instance segmentation).
2. **U-Net++** (مشفّر EfficientNet-B3): تدريب على **قصّات ROI** حول الجرح لتحسين حدود التجزئة.
3. **الاستدلال المدمج:** YOLO يحدد الصندوق → قص الـ ROI → U-Net++ يصحح القناع → إعادة رسم القناع على الإحداثيات الأصلية وحساب المساحة التقريبية.

لا يوجد تدريب مشترك بين النموذجين؛ الدمج يحدث فقط عند `combined` أو `predict_single_image`.

---

## 2. الإعدادات (`config.yaml`)

| المفتاح | المعنى |
|---------|--------|
| `seed` | تثبيت العشوائية |
| `data_root` | مجلد الجذر للصور (مثل `data/wound_focus_clean`) |
| `ann_train` / `ann_val` / `ann_test` | مسارات JSON بصيغة COCO لـ **فئة الجرح فقط** |
| `num_workers` | عدد عمال DataLoader (0 مناسب لـ Windows) |
| قسم `yolo` | نموذج Ultralytics، حجم الصورة، الدفعات، العصور، التعزيز، `perspective: 0` لحماية هندسة المقياس |
| قسم `unet` | مشفّر، حجم إدخال ROI، خسارة BCE+Dice، `roi_padding`، جدولة التعلم |
| قسم `combined` | عتبات الثقة وقناع U-Net، `pixels_per_cm` لمساحة الجرح، عدد عينات التصور |

**ملاحظة:** التجربة تستخدم فئة **`wound`** فقط (`WOUND_ONLY_CLASSES` في `pipeline_utils.py`). تصنيف العدوى **لا** يُدخل كمدخل تدريب في هذا المسار؛ يُستنتج أحياناً من اسم الملف في دوال الاستدلال (`-not-`).

---

## 3. المرحلة: `convert` (تحضير YOLO)

**الأمر:** `python train_model.py --stage convert`

**ما يحدث:**

1. **`prepare_yolo_dataset`** (`pipeline_utils.py`):
   - لكل تقسيم (train / val / test) يقرأ ملف COCO المحدد في `config`.
   - **`coco_to_yolo_seg`:** يستخرج مضلعات `segmentation` للفئات المطابقة لـ `wound`، يحوّل الإحداثيات إلى **0–1**، ويكتب سطراً لكل مضلع:  
     `class_id x1 y1 x2 y2 ...`
   - **تخزين التسميات:** الملفات تُكتب تحت **`data/wound_focus_clean/labels/`** (بجانب `images/`) بنفس اسم الجذر، لأن Ultralytics يطابق الصورة `.../images/name.jpg` مع `.../labels/name.txt`.
   - **`create_dataset_yaml`:** ينشئ `yolo_data/dataset.yaml` وملفات `train.txt`, `val.txt`, `test.txt` تحتوي على **مسارات مطلقة** لكل صورة.

2. **`validate_yolo_dataset`:** يتحقق من وجود القوائم وصور العينة وملفات `.txt` المقابلة.

**مخرجات:** `yolo_data/dataset.yaml`, `yolo_data/*.txt`, ومجلد `labels/` تحت `data_root`.

---

## 4. المرحلة: `yolo` (تدريب وتقييم YOLO)

**الأمر:** `python train_model.py --stage yolo`

**ما يحدث بالترتيب:**

1. إن لم يوجد `yolo_data/dataset.yaml`، يُستدعى التحويل تلقائياً.
2. **`train_yolo`:** يحمّل أوزان `yolo11m-seg.pt`، يستدعي `model.train(...)` من Ultralytics مع المعاملات من `config` (SGD، تعزيز، إلخ). النتائج تُحفظ تحت `checkpoints/yolo/train/` ثم تُنسخ `best.pt` و`last.pt` إلى `checkpoints/yolo/`.
3. **`evaluate_yolo`:** تشغيل `model.val(split="test")` وحفظ مقاييس الصندوق والتجزئة في `results/yolo/test_metrics.json`.
4. **`predict_yolo`:** عيّنة من صور الاختبار، حفظ صور `pred_*.png` في `results/yolo/predictions/`.

---

## 5. المرحلة: `unet` (تدريب U-Net++)

**الأمر:** `python train_model.py --stage unet`

**مجموعة البيانات — `WoundROIDataset`:**

- لكل **annotation** في COCO لفئة `wound`:
  - تحميل الصورة من `data_root / file_name`.
  - بناء قناع ثنائي من `segmentation` بالمضلعات.
  - قص **ROI** حسب `bbox` مع توسيع `roi_padding`.
  - تغيير الحجم إلى `input_size` (افتراضياً 256×256) وتطبيق Albumentations (تدريب/تحقق).

**التدريب:**

- نموذج **`segmentation_models_pytorch.UnetPlusPlus`** (مشفّر EfficientNet-B3، صنف واحد، `activation=None`).
- خسارة **`BCEDiceLoss`** (BCE + Dice بأوزان من `config`).
- محسن **AdamW** وجدولة **CosineAnnealingLR**.
- حفظ **`best_model.pth`** عند أعلى Dice على التحقق، و**`last_checkpoint.pth`** كل عصر.
- **إيقاف مبكر** إذا لم يتحسن Dice لعدد `early_stop_patience` عصور.
- بعد الانتهاء: تقييم على **test** بحساب Dice وIoU ودقة البكسل.

**مخرجات:** `checkpoints/unet/`, `results/unet/training_history.json`, `metrics_summary.json`, `unet_training_curves.png`.

---

## 6. المرحلة: `combined` (استدلال مدمج + تقييم)

**الأمر:** `python train_model.py --stage combined`

**`combined_inference` (لكل صورة):**

1. YOLO على المسار الكامل بعامل ثقة `yolo_conf_thresh`.
2. لكل صندوق مكتشف: توسيع الـ ROI بـ `roi_padding`، قص، تغيير الحجم إلى حجم U-Net، تطبيع ImageNet، تمرير U-Net++، ثنائية القناع بـ `unet_mask_thresh`.
3. رفع القناع من حجم الـ crop إلى الإحداثيات الكاملة ودمج القنوع عند عدة اكتشافات.

**`evaluate_combined`:**

- يمرّ على صور **test** في `ann_test`.
- يبني قناع GT من مضلعات COCO.
- يدمج قنوع التنبؤ ويحسب **Dice** و**IoU** مقابل GT.
- **`calculate_wound_area`:** مساحة تقريبية بالـ cm² من `pixels_per_cm` (قيمة تجريبية في الإعدادات).
- يحفظ `results/combined/metrics_summary.json`, `wound_areas.json`, وصوراً متراكبة في `results/combined/predictions/`.

**`predict_single_image`:** يغلف الاستدلال المدمج مع تقدير مساحة وقراءة **infection** من اسم الملف (للعرض فقط، ليس تدريباً).

---

## 7. ما بعد أي مرحلة: التقرير والملخص الشامل

عند نهاية `main()` (أي تشغيل لمرحلة أو `--stage all`):

- **`save_global_metrics_summary`:** يكتب `results/metrics_summary.json` يجمع yolo / unet / combined.
- **`generate_report`:** يكتب `reports/training_report.md` من جداول الإعدادات والمقاييس.

---

## 8. المرجع السريع للأوامر

```bash
cd experiments/YOLO11m_UNetPP
python train_model.py --stage convert    # COCO → YOLO + dataset.yaml
python train_model.py --stage yolo       # تدريب YOLO + val على test + عينات تنبؤ
python train_model.py --stage unet       # تدريب U-Net++ على ROI
python train_model.py --stage combined   # تقييم مسار YOLO+U-Net على test
python train_model.py --stage all        # كل المراحل بالترتيب
```

---

## 9. قيود بحثية

- جودة التسميات: طبقة **wound** فقط؛ الفئات الفرعية (أنسجة، إلخ) غير مستخدمة هنا.
- الاستخدام بحثي/تعليمي وليس أداة سريرية معتمدة.
- مساحة الجرح بالـ cm² تعتمد على **`pixels_per_cm`**؛ للاستخدام الدقيق يلزم معايرة مع **مقياس 3×3 سم** من المشروع الأوسع.

---

*آخر تحديث للتوثيق يتبع بنية `train_model.py` و`pipeline_utils.py` في المستودع.*
