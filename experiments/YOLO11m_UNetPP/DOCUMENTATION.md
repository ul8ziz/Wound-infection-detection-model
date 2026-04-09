# توثيق تجربة YOLO11m + U-Net++

هذا الملف يشرح **كل مرحلة تشغيل** و**كل عملية** في مجلد `experiments/YOLO11m_UNetPP`: من تحضير البيانات إلى التدريب والتقييم والاستدلال المدمج، بالإضافة إلى **التحسينات المُطبّقة** على النسخة الأصلية.

**الملفات الأساسية:**

| الملف | الدور |
|--------|--------|
| `config.yaml` | جميع المعاملات (بيانات، YOLO، U-Net++، الاستدلال المدمج) |
| `pipeline_utils.py` | تحويل COCO→YOLO، مجموعات بيانات U-Net++ و`WoundDataset`، تحويلات Albumentations |
| `train_model.py` | واجهة CLI لجميع المراحل، التدريب، التقييم، التقرير |
| `training_pipeline.ipynb` | نفس المنطق بشكل تفاعلي |
| `augment_offline.py` | تعزيز البيانات دون اتصال (Offline Augmentation) لتوسيع مجموعة التدريب |
| `build_wound_marker_dataset.py` | بناء تسميات COCO تشمل الجرح + المقياس (SizeMarker) |

**البيانات:** مأخوذة من جذر المشروع، لا تُنسخ داخل مجلد التجربة. المسارات الافتراضية في `config.yaml` تشير إلى `data/wound_focus_clean` وملفات COCO *wound-only*.

---

## 1. مبدأ التجربة (معماريتان مستقلتان ثم دمج)

1. **YOLO11m-seg:** كشف الجرح على صورة كاميرة + قناع تجزئة أولي (instance segmentation).
2. **U-Net++** (مشفّر من `config.yaml`، افتراضياً **EfficientNet-B1**): تدريب على **قصّات ROI** حول الجرح لتحسين حدود التجزئة.
3. **الاستدلال المدمج:** YOLO يحدد الصندوق → قص الـ ROI → U-Net++ يصحح القناع (مع TTA) → NMS على القنوع → إعادة رسم القناع على الإحداثيات الأصلية → معايرة المساحة عبر المقياس → تصنيف العدوى.
4. **تصنيف العدوى:** مصنّف MLP خفيف يعمل على خصائص لونية/نسيجية من منطقة الجرح.

لا يوجد تدريب مشترك بين النموذجين؛ الدمج يحدث فقط عند `combined` أو `predict_single_image`.

---

## 2. الإعدادات (`config.yaml`)

| المفتاح | المعنى |
|---------|--------|
| `seed` | تثبيت العشوائية |
| `data_root` | مجلد الجذر للصور (مثل `data/wound_focus_clean`) |
| `ann_train` / `ann_val` / `ann_test` | مسارات JSON بصيغة COCO |
| `num_workers` | عدد عمال DataLoader (0 مناسب لـ Windows) |
| قسم `yolo` | نموذج Ultralytics، حجم الصورة **1024**، الدفعات **4**، التعزيز المُحسّن |
| قسم `unet` | مشفّر، حجم إدخال ROI (افتراضياً **256×256** لتسريع التدريب؛ يمكن **384×384** للجودة)، خسارة **Focal+Dice**، `roi_padding`، جدولة التعلم |
| قسم `combined` | عتبة ثقة **0.25**، `pixels_per_cm`، `marker_real_cm` لمعايرة المقياس |

**إعداد سريع مقابل إعداد جودة (U-Net++):** القيم الافتراضية في `config.yaml` تقلّل **زمن التدريب** (`input_size` أصغر، مشفّر أخف، عصور أقل، `early_stop_patience` أصغر). لأقصى تفصيل للحدود كما في تجارب التحسين (م5 / ج3)، استخدم تقريباً: `encoder: efficientnet-b3`، `input_size: [384, 384]`، `epochs: 50`، `scheduler_T_max: 50`، `early_stop_patience: 10`.

---

## 3. المرحلة: `convert` (تحضير YOLO)

**الأمر:** `python train_model.py --stage convert`

**ما يحدث:**

1. **`prepare_yolo_dataset`** (`pipeline_utils.py`):
   - لكل تقسيم (train / val / test) يقرأ ملف COCO المحدد في `config`.
   - **`coco_to_yolo_seg`:** يستخرج مضلعات `segmentation` للفئات المطابقة، يحوّل الإحداثيات إلى **0–1**، ويكتب سطراً لكل مضلع:  
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
2. **`train_yolo`:** يحمّل أوزان `yolo11m-seg.pt`، يستدعي `model.train(...)` من Ultralytics مع المعاملات من `config` (SGD، تعزيز مُحسّن، إلخ). النتائج تُحفظ تحت `checkpoints/yolo/train/` ثم تُنسخ `best.pt` و`last.pt` إلى `checkpoints/yolo/`.
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
  - يمكن تشغيل أوضاع قص متعددة لتقليل فجوة التدريب/الاستدلال:
    - `gt_only`: قص GT التقليدي (السلوك القديم).
    - `mixed`: مزيج من GT + jitter + صناديق YOLO المخزنة مسبقاً.
    - `yolo_predicted`: استخدام صناديق YOLO المخزنة إذا توفرت.
  - تغيير الحجم إلى `input_size` من `config` (يدعم الآن **256×256** و **384×384** و **512×512**) وتطبيق Albumentations المُحسّن.

**التدريب:**

- نموذج التقسيم من `segmentation_models_pytorch`:
  - `unetplusplus` (الخط الأساسي).
  - `deeplabv3plus` للمقارنة الأقوى إذا لزم.
- الخسائر المتاحة:
  - `focal_dice` (السلوك القديم).
  - `focal_dice_boundary` = Focal + Dice + حدّ boundary-aware إضافي لتحسين الحواف الدقيقة.
- محسن **AdamW** وجدولة **CosineAnnealingLR**.
- يدعم `resume_checkpoint` و `freeze_encoder` للفين-تيون السريع.
- حفظ **`best_model.pth`** عند أعلى Dice على التحقق، و**`last_checkpoint.pth`** كل عصر.
- **إيقاف مبكر** إذا لم يتحسن Dice لعدد `early_stop_patience` عصور.
- بعد الانتهاء: تقييم على **test** بحساب Dice وIoU ودقة البكسل.
- عند تعيين `experiment_name` تُحفظ النتائج في مجلدات معزولة مثل:
  - `checkpoints/unet/<experiment_name>/`
  - `results/unet/<experiment_name>/`

**مخرجات:** `checkpoints/unet/`, `results/unet/training_history.json`, `metrics_summary.json`, `unet_training_curves.png`.

---

## 6. المرحلة: `combined` (استدلال مدمج + تقييم)

**الأمر:** `python train_model.py --stage combined`

**`combined_inference` (لكل صورة):**

1. YOLO على المسار الكامل بعامل ثقة `yolo_conf_thresh` (0.25).
2. تصفية الاكتشافات: فقط فئة **wound** (class_id=0) تُمرر لـ U-Net++.
3. لكل صندوق wound: توسيع الـ ROI بـ `roi_padding`، قص، تغيير الحجم، تطبيع ImageNet.
4. **TTA (Test-Time Augmentation):** تمرير الصورة الأصلية + المعكوسة أفقياً عبر U-Net++، ثم متوسط الاحتمالات قبل العتبة.
5. يدعم **multi-scale refinement** عبر `multi_scale_roi_paddings` و `multi_scale_fusion`:
   - مثال: tight ROI + pad 10% + pad 20% ثم دمج احتمالات القناع.
6. يدعم **refinement_postprocess** الاختياري بعد التنبؤ، مثل `boundary_refine`.
7. **Mask NMS:** إزالة القنوع المتداخلة (IoU > 0.5) للحد من التكرار.
8. **معايرة المقياس:** إذا اكتشف YOLO مقياس 3×3 سم (class_id=1)، تُحسب `pixels_per_cm` تلقائياً بدلاً من القيمة الثابتة.

**`evaluate_combined`:**

- يمرّ على صور **test** في `ann_test`.
- يبني قناع GT من مضلعات COCO.
- يدمج قنوع التنبؤ ويحسب **Dice** و**IoU** مقابل GT.
- **تقييم COCO AP:** يحوّل القنوع إلى RLE ويحسب AP/AP50/AP75 عبر `pycocotools` للمقارنة المنصفة مع Mask R-CNN.
- يحسب الآن **`coco_combined_AP50`** و **`coco_combined_AP75`**.
- **`calculate_wound_area`:** مساحة بالـ cm² باستخدام `pixels_per_cm` المُعايرة أو الافتراضية.
- عند تعيين `experiment_name` تُكتب نتائج combined داخل `results/combined/<experiment_name>/`.

---

## 7. المرحلة: `infection` (تصنيف العدوى)

**الأمر:** `python train_model.py --stage infection`

**ما يحدث:**

1. تحميل نماذج YOLO و U-Net++ المدرّبة.
2. لكل صورة تدريب/تحقق: تشغيل الاستدلال المدمج واستخراج **15 خاصية** من منطقة الجرح:
   - متوسط وانحراف RGB (6 قيم)
   - متوسط وانحراف HSV (6 قيم)
   - نسبة مساحة الجرح، نسبة المحيط، الانضغاطية (3 قيم)
3. تسمية العدوى من اسم الملف (`_infected` / `_not_infected`).
4. تدريب **`WoundInfectionClassifier`** (MLP: 15→64→32→1) مع BCEWithLogitsLoss ووزن متوازن للفئات.
5. حفظ النموذج مع إحصائيات التطبيع في `checkpoints/infection/infection_classifier.pth`.
6. حفظ المقاييس (دقة، استدعاء، F1) في `results/infection/metrics_summary.json`.

**الاستخدام عند الاستدلال:**

- `predict_infection()` تُستدعى في `predict_single_image()` لإضافة تصنيف العدوى التلقائي.
- النتيجة تظهر على الصورة المتراكبة: `infected` أو `non_infected` مع احتمال.

---

## 8. ما بعد أي مرحلة: التقرير والملخص الشامل

عند نهاية `main()` (أي تشغيل لمرحلة أو `--stage all`):

- **`save_global_metrics_summary`:** يكتب `results/metrics_summary.json` يجمع yolo / unet / combined.
- **`generate_report`:** يكتب `reports/training_report.md` من جداول الإعدادات والمقاييس (يشمل نتائج العدوى إن وُجدت).

---

## 9. المرجع السريع للأوامر

```bash
cd experiments/YOLO11m_UNetPP

# المراحل الأساسية
python train_model.py --stage convert    # COCO → YOLO + dataset.yaml
python train_model.py --stage yolo       # تدريب YOLO + تقييم على test
python train_model.py --stage unet       # تدريب U-Net++ على ROI
python train_model.py --stage combined   # تقييم مسار YOLO+U-Net على test
python train_model.py --stage infection  # تدريب مصنّف العدوى
python train_model.py --stage all        # كل المراحل بالترتيب

# أدوات إضافية
python augment_offline.py                        # توسيع مجموعة التدريب ×4
python augment_offline.py --num-augments 4       # عدد نسخ مُعزّزة مخصص
python build_wound_marker_dataset.py             # بناء تسميات wound+marker
```

---

## 10. قيود بحثية

- الاستخدام بحثي/تعليمي وليس أداة سريرية معتمدة.
- تصنيف العدوى يعتمد على خصائص لونية/نسيجية بسيطة؛ ليس بديلاً عن التشخيص السريري.
- عند استخدام فئة `wound` فقط: `pixels_per_cm` ثابت (26.0). لدقة أعلى شغّل `build_wound_marker_dataset.py` لتفعيل اكتشاف المقياس.

---
---

# توثيق التحسينات

هذا القسم يوثّق **جميع التحسينات** المُطبّقة على التجربة مقارنة بالنسخة الأصلية (baseline)، مع شرح **المشكلة** التي يحلّها كل تحسين، و**التغيير التقني**، و**الأثر المتوقع**.

---

## نتائج النسخة الأصلية (Baseline)

```
                            التحقق (عصر 75)    الاختبار
YOLO bbox mAP50             0.868              0.764
YOLO segm mAP50             0.727              0.623
YOLO segm mAP50-95          0.215              0.235
U-Net++ Dice                0.774              0.768
U-Net++ IoU                 0.647              0.637
المسار المدمج Dice           --                 0.680
المسار المدمج IoU            --                 0.551
```

مرجع Mask R-CNN: bbox mAP50 = 0.398، segm mAP50 = 0.217.

---

## المشكلات المُكتشفة (مرتّبة حسب الأهمية)

### م1: انخفاض Dice المدمج من 0.768 إلى 0.680

Dice لـ U-Net++ المنفرد على الاختبار **0.768**، لكن المسار المدمج ينخفض إلى **0.680** (انخفاض 11.5%). الأسباب:
- **YOLO يفوّت ~30% من الجروح**: استدعاء القنوع (recall) = 69.4% فقط.
- **5 صور اختبار لم تحصل على أي اكتشاف** (50 صورة تم تقييمها من أصل 55).
- عتبة الثقة `yolo_conf_thresh = 0.5` مرتفعة جداً لمهمة طبية حساسة للاستدعاء.

### م2: حساب مساحة الجرح بقيمة ثابتة `pixels_per_cm = 26.0`

- المسار **لا يكتشف المقياس 3×3 سم**، فلا توجد معايرة لكل صورة.
- نتائج غير واقعية: **720 سم²**، **689 سم²**، **587 سم²**.

### م3: مجموعة بيانات صغيرة بدون تعزيز خارجي

- **257 صورة تدريب** فقط — قليل لمهمة كشف + تجزئة.
- YOLO يستخدم تعزيزاً آنياً (mosaic, mixup) لكن U-Net++ يحصل على تحويلات أساسية فقط.

### م4: جودة تجزئة YOLO خشنة (mAP50-95 = 0.235)

- الفجوة بين bbox mAP50-95 (0.494) و segm mAP50-95 (0.235) كبيرة جداً.
- عند `imgsz=640`، تفاصيل حدود الجرح الدقيقة تضيع.

### م5: استقرار U-Net++ مبكراً (Dice عالق حول 0.77)

- Dice يصل 0.77 بالعصر 6 ولا يتحسن كثيراً بعدها.
- حجم إدخال 256×256 صغير لالتقاط تفاصيل نسيج الجرح.
- التعزيز أساسي جداً (بدون elastic، بدون CLAHE).

### م6: لا يوجد تصنيف عدوى

- تسميات العدوى موجودة في أسماء الملفات لكنها **غير مستخدمة** في أي مرحلة تدريب.

### م7: فجوة overfitting بين التحقق والاختبار

- YOLO segm mAP50: تحقق = 0.727 مقابل اختبار = 0.623 (فجوة 14%).
- `mosaic=1.0` عدواني جداً لمجموعة بيانات طبية صغيرة.

---

## التحسينات المُطبّقة

### تحسين أ1: خفض عتبة ثقة YOLO (يحلّ م1)

| البند | القيمة القديمة | القيمة الجديدة |
|-------|---------------|---------------|
| `combined.yolo_conf_thresh` | 0.5 | **0.25** |

**السبب:** عتبة 0.5 تفوّت ~30% من الجروح. خفضها إلى 0.25 يستعيد اكتشافات كانت مفقودة. عتبة قناع U-Net++ (0.5) لا تزال تصفي الإيجابيات الكاذبة.

**الملف:** `config.yaml` سطر 62.

**الأثر المتوقع:** ارتفاع Dice المدمج من ~0.68 إلى ~0.72.

---

### تحسين أ2: تعزيز وقت الاختبار TTA (يحلّ م1، م5)

**ما أُضيف:** دالة `_unet_predict_with_tta()` في `train_model.py`.

**الآلية:**
1. تمرير قصّة ROI الأصلية عبر U-Net++ ← احتمالات sigmoid.
2. عكس القصّة أفقياً ← تمرير ← عكس الناتج.
3. متوسط الاحتمالين ← عتبة ← القناع النهائي.

**السبب:** TTA يحسّن دقة الحدود مجاناً بدون إعادة تدريب.

**الأثر المتوقع:** +1-3% Dice إضافي.

---

### تحسين أ3: NMS على القنوع المُحسّنة (يحلّ م1)

**ما أُضيف:** دالة `_mask_nms()` في `train_model.py`.

**الآلية:** عند وجود عدة اكتشافات YOLO لنفس المنطقة، تُحسب IoU بين القنوع المُحسّنة وتُزال المتداخلة (IoU > 0.5) مع الاحتفاظ بالأعلى ثقة.

**السبب:** بعد خفض عتبة الثقة، قد تظهر اكتشافات مكررة — NMS يمنع حساب المساحة مرتين.

---

### تحسين ب1: تعزيزات U-Net++ أقوى (يحلّ م5)

**ما تغيّر:** دالة `get_unet_transforms()` في `pipeline_utils.py`.

**التحويلات المُضافة للتدريب:**

| التحويل | المعاملات | الاحتمال |
|---------|----------|---------|
| `ShiftScaleRotate` | shift=0.1، scale=0.15، rotate=15° | 0.5 |
| `CLAHE` | clip_limit=2.0 | 0.3 |
| `ColorJitter` | brightness/contrast=0.2، saturation=0.1، hue=0.05 | 0.3 |
| `ElasticTransform` | alpha=50، sigma=10 (خفيف) | 0.1 |

**السبب:** التعزيز الأساسي (flips + brightness فقط) غير كافٍ لـ 257 صورة. هذه التحويلات تزيد التنوع مع الحفاظ على الواقعية الطبية.

**الأثر المتوقع:** +2-4% Dice.

---

### تحسين ب2: تعزيز خارجي ×4 لمجموعة البيانات (يحلّ م3)

**ما أُنشئ:** سكريبت `augment_offline.py`.

**الآلية:**
1. يقرأ `train_wound_only.json` مع الصور الأصلية.
2. لكل صورة: ينشئ 3 نسخ مُعزّزة (تحويلات طبية آمنة).
3. يحوّل القنوع إلى مضلعات ويُحدّث بيانات COCO.
4. المخرج: `data/wound_focus_clean/augmented/train_augmented.json` + صور.

**التعزيزات المُستخدمة:** HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast, CLAHE, ColorJitter, GaussNoise, GaussianBlur.

**الاستخدام:**
```bash
python augment_offline.py                    # 3 نسخ = 257×4 ≈ 1028 صورة
python augment_offline.py --num-augments 4   # 4 نسخ = 257×5 ≈ 1285 صورة
```

ثم تحديث `config.yaml`:
```yaml
ann_train: "data/wound_focus_clean/augmented/train_augmented.json"
data_root: "data/wound_focus_clean/augmented"
```

**الأثر المتوقع:** تحسين mAP وDice بفضل زيادة بيانات التدريب 4 أضعاف.

---

### تحسين ج1: رفع حجم صورة YOLO إلى 1024 (يحلّ م4)

| البند | القيمة القديمة | القيمة الجديدة |
|-------|---------------|---------------|
| `yolo.image_size` | 640 | **1024** |
| `yolo.batch_size` | 8 | **4** |

**السبب:** عند 640 بكسل، تفاصيل حدود الجرح الدقيقة تضيع وsegm mAP50-95 ضعيف (0.235). رفع الدقة إلى 1024 يحسّن جودة القنوع بشكل ملحوظ.

**الأثر المتوقع:** +5-10% في segm mAP50-95.

---

### تحسين ج2: ضبط تعزيز YOLO للبيانات الطبية (يحلّ م7)

| البند | القيمة القديمة | القيمة الجديدة | السبب |
|-------|---------------|---------------|-------|
| `mosaic` | 1.0 | **0.5** | Mosaic عدواني جداً لـ 257 صورة |
| `mixup` | 0.1 | **0.0** | خلط الصور غير مناسب طبياً |
| `close_mosaic` | 10 (افتراضي) | **15** | إيقاف mosaic أبكر للتعلم النظيف |

**الأثر المتوقع:** تقليل فجوة val-test (overfitting) بنسبة 5-8%.

---

### تحسين ج3: رفع حجم إدخال U-Net++ إلى 384 (يحلّ م5)

| البند | القيمة القديمة | القيمة الجديدة (إعداد الجودة) |
|-------|---------------|---------------|
| `unet.input_size` | [256, 256] | **[384, 384]** |

**السبب:** حجم 256 صغير لالتقاط تفاصيل نسيج الجرح. EfficientNet-B3 يعمل بكفاءة على 384.

**الأثر المتوقع:** +2-4% Dice.

**الوضع الحالي في المستودع:** الافتراضي في `config.yaml` عاد إلى **[256, 256]** مع مشفّر أخف (**b1**) لتقليل **زمن التدريب**. لإعادة إعداد ج3 بالكامل، اضبط أيضاً `encoder: efficientnet-b3` والعصور/`scheduler_T_max`/`early_stop_patience` كما في تعليقات `config.yaml`.

---

### تحسين ج4: استبدال BCE+Dice بـ Focal+Dice (يحلّ م5)

**ما أُضيف:** فئتا `FocalLoss` و`FocalDiceLoss` في `train_model.py`.

| البند | القيمة القديمة | القيمة الجديدة |
|-------|---------------|---------------|
| `unet.loss_type` | (غير موجود) | **"focal_dice"** |
| `unet.focal_alpha` | -- | **0.25** |
| `unet.focal_gamma` | -- | **2.0** |

**الآلية:**
- **Focal Loss:** يركّز على البكسلات الصعبة (حدود الجرح) بدلاً من معاملة جميع البكسلات بالتساوي.
- **الصيغة:** `loss = alpha * (1 - p_t)^gamma * BCE`
- مدمج مع Dice Loss بنسبة 50/50.

**التوافق:** الفئة القديمة `BCEDiceLoss` لا تزال موجودة لتوافقية نقاط الحفظ. يمكن التبديل عبر `loss_type: "bce_dice"`.

---

### تحسين د1: معايرة المقياس (SizeMarker) (يحلّ م2)

**ما أُنشئ:** سكريبت `build_wound_marker_dataset.py` + دالة `calculate_pixels_per_cm_from_marker()`.

**الآلية:**

1. **بناء البيانات:** السكريبت يقرأ `annotations_cleaned.json` الأصلي ويستخرج فئتي:
   - `wound` (ВсяРана، id=1 أصلي → id=1 جديد)
   - `marker` (Метка для размерности، id=2 أصلي → id=2 جديد)
2. ينشئ `train/val/test_wound_marker.json` تحت `data/wound_focus_clean/`.

3. **المعايرة في الاستدلال:**
   - إذا اكتشف YOLO مقياساً (class_id=1): يحسب `pixels_per_cm` من متوسط أبعاد الصندوق مقسوماً على `marker_real_cm` (3 سم).
   - إذا لم يُكتشف: تُستخدم القيمة الافتراضية (26.0).
   - النتيجة تُضاف إلى `wound_areas.json` مع حقل `marker_detected`.

**الاستخدام:**
```bash
python build_wound_marker_dataset.py
# ثم تحديث config.yaml:
# ann_train: "data/wound_focus_clean/train_wound_marker.json"
# ann_val:   "data/wound_focus_clean/val_wound_marker.json"
# ann_test:  "data/wound_focus_clean/test_wound_marker.json"
```

---

### تحسين د2: تصنيف العدوى (يحلّ م6)

**ما أُضيف:** فئة `WoundInfectionClassifier` ودوال مساعدة في `train_model.py`.

**المعمارية:**
```
استخراج 15 خاصية → تطبيع → MLP (15→64→32→1) → sigmoid → infected/non_infected
```

**الخصائص المُستخرجة (15 بُعد):**

| المجموعة | الخصائص |
|---------|---------|
| RGB | متوسط R, G, B + انحراف R, G, B |
| HSV | متوسط H, S, V + انحراف H, S, V |
| شكلية | نسبة مساحة الجرح، نسبة المحيط، الانضغاطية |

**التدريب:**
- تسميات من أسماء الملفات: `_infected` → 1، `_not_infected` → 0.
- BCEWithLogitsLoss مع وزن متوازن للفئات (158 مصاب / 222 غير مصاب).
- 200 عصر على جميع العينات.
- حفظ النموذج + إحصائيات التطبيع في `checkpoints/infection/`.

**المرحلة:** `python train_model.py --stage infection`

---

### تحسين هـ1: تقييم COCO AP الموحّد (يحلّ المقارنة غير المنصفة)

**ما أُضيف:** دالة `evaluate_combined_coco()` في `train_model.py`.

**الآلية:**
1. تحويل قنوع التنبؤ إلى RLE عبر `pycocotools.mask`.
2. بناء قائمة نتائج COCO (bbox + segm).
3. تقييم عبر `COCOeval` لحساب AP, AP50, AP75.

**المقاييس المحفوظة في `results/combined/coco_metrics.json`:**

| المقياس | المعنى |
|---------|--------|
| `coco_bbox_AP` | AP للصناديق عند IoU 0.50:0.95 |
| `coco_bbox_AP50` | AP للصناديق عند IoU 0.50 |
| `coco_bbox_AP75` | AP للصناديق عند IoU 0.75 |
| `coco_segm_AP` | AP للقنوع عند IoU 0.50:0.95 |
| `coco_segm_AP50` | AP للقنوع عند IoU 0.50 |
| `coco_segm_AP75` | AP للقنوع عند IoU 0.75 |
| `coco_combined_AP50` | متوسط bbox + segm AP50 |

**الفائدة:** مقارنة منصفة ومباشرة مع تجربة Mask R-CNN (نفس المقاييس، نفس بروتوكول التقييم).

---

## ملخص التغييرات على الملفات

| الملف | التغييرات |
|-------|----------|
| `config.yaml` | 12 معامل مُحدّث (imgsz, batch, mosaic, mixup, close_mosaic, input_size, conf_thresh, loss_type, focal params, marker_real_cm) |
| `pipeline_utils.py` | تعزيزات U-Net++ أقوى (4 تحويلات جديدة)، ثابت `WOUND_MARKER_CLASSES` |
| `train_model.py` | TTA، Mask NMS، FocalLoss، FocalDiceLoss، معايرة المقياس، تقييم COCO AP، مصنّف العدوى الكامل، مرحلة `infection` جديدة |
| `augment_offline.py` | **ملف جديد** — توسيع البيانات ×4 |
| `build_wound_marker_dataset.py` | **ملف جديد** — بناء تسميات wound+marker |

---

## الأثر المتوقع (تراكمي)

| التحسين | Dice المدمج المتوقع | الجهد |
|---------|---------------------|-------|
| النسخة الأصلية | 0.680 | -- |
| خفض عتبة الثقة (أ1) | ~0.72 | 1 دقيقة (تعديل config) |
| TTA (أ2) | ~0.74 | مُطبّق |
| تعزيزات أقوى (ب1) | ~0.76 | إعادة تدريب |
| ضبط تعزيز YOLO (ج2) | ~0.78 | إعادة تدريب |
| تعزيز خارجي ×4 (ب2) | ~0.80 | تشغيل السكريبت + إعادة تدريب |
| أحجام إدخال أكبر (ج1+ج3) | ~0.82 | إعادة تدريب |
| المقياس (د1) | دقة المساحة | تشغيل السكريبت |
| Focal+Dice (ج4) | +1-2% إضافي | إعادة تدريب |
| تصنيف العدوى (د2) | مقياس جديد | مُطبّق |

---

## خطوات التشغيل المُوصى بها

```bash
cd experiments/YOLO11m_UNetPP

# 1. (اختياري) توسيع البيانات
python augment_offline.py
# ثم تحديث ann_train و data_root في config.yaml

# 2. (اختياري) إضافة فئة المقياس
python build_wound_marker_dataset.py
# ثم تحديث ann_train/val/test في config.yaml

# 3. التدريب الكامل
python train_model.py --stage all
```

---

*آخر تحديث: أبريل 2026 — يتبع بنية `train_model.py` و`pipeline_utils.py` المُحسّنة.*
