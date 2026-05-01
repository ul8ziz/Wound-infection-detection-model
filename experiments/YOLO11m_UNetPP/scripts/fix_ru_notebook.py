"""Fix training_pipeline_ru.ipynb:
1. Clear all stale outputs
2. Fix YOLO model loading (prefer best.pt over deleted yolo11m-seg.pt)
3. Add final results summary cell in Russian
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "training_pipeline_ru.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── 1. Clear ALL stale outputs ─────────────────────────────────────────────
cleared = 0
for c in nb["cells"]:
    if c["cell_type"] == "code" and c.get("outputs"):
        c["outputs"] = []
        c["execution_count"] = None
        cleared += 1
print(f"Cleared outputs in {cleared} cells")

# ── 2. Fix YOLO loading in every code cell ─────────────────────────────────
OLD_YOLO_LINE = 'yolo_model = build_yolo_model(CONFIG["yolo"].get("model", "yolo11m-seg.pt"))'

NEW_YOLO_LINES = (
    '_yolo_trained = SCRIPT_DIR / "checkpoints" / "yolo" / "best.pt"\n'
    '_yolo_weights = str(_yolo_trained) if _yolo_trained.exists() else CONFIG["yolo"].get("model", "yolo11m-seg.pt")\n'
    'yolo_model = build_yolo_model(_yolo_weights)'
)

# Indented version (inside if-block)
OLD_YOLO_INDENT = '    yolo_model = build_yolo_model(CONFIG["yolo"].get("model", "yolo11m-seg.pt"))'
NEW_YOLO_INDENT = (
    '    _yolo_trained = SCRIPT_DIR / "checkpoints" / "yolo" / "best.pt"\n'
    '    _yolo_weights = str(_yolo_trained) if _yolo_trained.exists() else CONFIG["yolo"].get("model", "yolo11m-seg.pt")\n'
    '    yolo_model = build_yolo_model(_yolo_weights)'
)

fixed = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    new_src = src.replace(OLD_YOLO_INDENT, NEW_YOLO_INDENT)
    new_src = new_src.replace(OLD_YOLO_LINE, NEW_YOLO_LINES)
    if new_src != src:
        c["source"] = [new_src]
        fixed += 1
        print(f"  Cell {i}: YOLO loading fixed")

print(f"Fixed YOLO loading in {fixed} cells")

# ── 3. Remove duplicate summary cell if already present ───────────────────
nb["cells"] = [c for c in nb["cells"]
               if "Итоговые результаты" not in "".join(c.get("source", []))]

# ── 4. Add final results summary cell (Russian) ────────────────────────────
SUMMARY_RU = """\
## 6: Итоговые результаты — Кампания по улучшению модели

### Лучшая конфигурация (текущий `config.yaml`)
| Параметр | Значение |
|---|---|
| Модель YOLO | YOLO11m-seg @ **1024 px** |
| Кодировщик U-Net++ | EfficientNet-B1 @ **512×512** |
| Функция потерь | FocalDiceBoundary (boundary_weight=0.20) |
| Отступ ROI (обучение + вывод) | **0.20** |
| Порог уверенности YOLO | **0.25** |
| TTA | включён |
| Многомасштабные отступы ROI | [0.10, 0.20, 0.30] |

---

### Достигнутые результаты — тестовый набор (55 изображений)

| Метрика | Базовая линия | **Лучший результат (phase7)** | Прирост |
|---|---|---|---|
| Mean Dice | 0.729 | **0.770** | +4.1 пп |
| Median Dice | ~0.780 | **0.843** | +6.3 пп |
| Mean IoU | 0.582 | **0.667** | +8.5 пп |
| COCO bbox AP50 | 0.778 | **0.792** | +1.4 пп |
| COCO bbox AP75 | 0.456 | **0.636** | **+18.0 пп** |
| COCO segm AP50 | 0.649 | **0.720** | +7.1 пп |
| COCO segm AP75 | 0.129 | **0.314** | **+18.5 пп** |
| COCO combined AP50 | 0.740 | **0.756** | +1.6 пп |
| COCO combined AP75 | 0.204 | **0.475** | **+27.1 пп** |

---

### Этапы улучшений

| Этап | Изменение | Δ Dice | Примечание |
|---|---|---|---|
| Базовая | — | 0.729 | Начальная точка |
| Этап 1 | Boundary-Aware Loss | 0.729 → 0.747 (+1.8) | segm_AP75 +11 пп |
| Этап 2 | 512×512 (с нуля) | 0.747 → 0.765 (+1.8) | Больше деталей контуров |
| Этап 3 | ElasticTransform + FancyPCA | 0.765 → 0.754 (−1.1) | **Отменён** — вредит малому датасету |
| Этап 4 | YOLO @ 1024 px | 0.760 (combined) | segm_AP75 +22.5 пп суммарно |
| Этап 5 | 5-кратная кросс-валидация (ансамбль) | segm_AP50 +1.4 пп | Dice без изменений |
| Этап 7 | Fine-tune + ROI 0.20 + порог 0.25 | **0.760 → 0.770 (+1.0)** | Лучший итоговый результат |

---

### Активные контрольные точки
- **YOLO**: `checkpoints/yolo/best.pt`
- **U-Net++**: `checkpoints/unet/best_model.pth` (phase7_finetune, эпоха 22, val_Dice=0.829)
- **K-Fold ансамбль**: `checkpoints/unet/kfold/fold_{0..4}/best_model.pth`
"""

nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [SUMMARY_RU],
})
print("Cell 12 (RU results summary) added")

# ── 5. Update Cell 0 description ──────────────────────────────────────────
NEW_INTRO = """\
## YOLO11m + Уточнение сегментации ROI

**Цель:**
Двухэтапный конвейер обработки ран со структурированными экспериментами по сегментации.

Этап 1: YOLO11m-seg @ **1024 px** обнаруживает ограничивающие рамки и грубые маски ран.

Этап 2: U-Net++ (EfficientNet-B1 @ **512×512**) уточняет маски на обрезанных областях интереса (ROI).

Этот ноутбук поддерживает:
- Обучение только на GT ROI / смешанный режим / кэш YOLO-ROI
- Перебор разрешений (256 / 384 / **512**)
- Функция потерь **FocalDiceBoundary** (boundary_weight=0.20)
- Многомасштабное TTA-уточнение в комбинированном конвейере

**Лучший достигнутый результат:**  
Mean Dice = **0.770** | segm_AP75 = **0.314** | combined_AP75 = **0.475**

**Набор данных:**
`data/wound_focus_clean/` с готовым разделением на train / val / test.
"""
nb["cells"][0]["source"] = [NEW_INTRO]
print("Cell 0: intro updated")

# ── Save ───────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"\nSaved: {NB_PATH}")
print(f"Total cells: {len(nb['cells'])}")
