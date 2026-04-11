# ResearchRabbit — Literature Guide for Wound Infection Detection Project

> **Purpose:** Provide seed papers and structured search queries for use in
> [ResearchRabbit](https://www.researchrabbit.ai/) to build a comprehensive
> reference network for a Master's thesis on postoperative wound infection
> detection using deep learning.
>
> **How to use:** Add the seed papers listed below to a Collection in
> ResearchRabbit, then explore connections (Similar Work, Citations, References)
> to automatically expand your reference base.

---

## Table of Contents

1. [Category 1: Wound Segmentation & Detection with Deep Learning](#1-wound-segmentation-and-detection)
2. [Category 2: YOLO for Medical Imaging](#2-yolo-for-medical-imaging)
3. [Category 3: U-Net & U-Net++ Architectures](#3-unet-and-unetpp-architectures)
4. [Category 4: Mask R-CNN for Medical Imaging](#4-mask-rcnn-for-medical-imaging)
5. [Category 5: Postoperative Wound Infection Detection](#5-postoperative-wound-infection-detection)
6. [Category 6: Data Augmentation in Medical Imaging](#6-medical-image-augmentation)
7. [Category 7: Transfer Learning in Medical Imaging](#7-transfer-learning-in-medical-imaging)
8. [Category 8: Cascaded Detection–Segmentation Pipelines](#8-cascaded-detection-segmentation-pipelines)
9. [Category 9: Computerized Wound Area Measurement](#9-computerized-wound-area-measurement)
10. [Category 10: Evaluation Metrics (COCO AP, Dice, IoU)](#10-evaluation-metrics)
11. [Category 11: EfficientNet & Efficient Architectures](#11-efficientnet-and-efficient-architectures)
12. [Category 12: Focal Loss & Dice Loss](#12-focal-loss-and-dice-loss)
13. [Suggested Search Queries for ResearchRabbit](#search-queries)
14. [Collection Building Plan in ResearchRabbit](#collection-building-plan)

---

<a id="1-wound-segmentation-and-detection"></a>
## 1. Wound Segmentation & Detection with Deep Learning

> **Project relevance:** This is the core topic — detecting and segmenting wounds in clinical photographs.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 1.1 | **Fully Automatic Wound Segmentation with Deep Convolutional Neural Networks** | Wang C. et al. | 2020 | `10.1038/s41598-020-68364-0` | One of the earliest works on fully automatic wound segmentation with deep CNNs |
| 1.2 | **Wound Segmentation with Dynamic Illumination Correction and Dual-View Semantic Fusion** | Oota S. et al. | 2023 | `10.1109/TMI.2023.3272710` | Advanced techniques for handling illumination challenges in wound images |
| 1.3 | **A Survey on Deep Learning for Skin Lesion Segmentation** | Mirikharaji Z. et al. | 2023 | `10.1016/j.media.2023.102863` | Comprehensive survey covering different architectures for skin lesion segmentation |
| 1.4 | **Deep Learning for Chronic Wound Image Analysis: A Comprehensive Review** | Anisuzzaman D.M. et al. | 2022 | `10.1016/j.compbiomed.2022.105616` | Thorough review of deep learning techniques for chronic wound image analysis |
| 1.5 | **WoundSeg: A Dataset and Benchmark for Wound Segmentation** | -- | 2023 | Search on Google Scholar | Dataset and benchmark for wound segmentation |

---

<a id="2-yolo-for-medical-imaging"></a>
## 2. YOLO for Medical Imaging

> **Project relevance:** Stage 1 of the cascaded pipeline uses YOLO11m-seg for wound detection and ROI extraction.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 2.1 | **YOLOv8 for Object Detection in Medical Imaging: A Comprehensive Review** | -- | 2024 | Search on Google Scholar | Review of modern YOLO variants in medical imaging |
| 2.2 | **Real-Time Object Detection for Medical Imaging: YOLO-Based Approaches** | -- | 2023 | Search on arXiv | YOLO applications in real-time medical detection |
| 2.3 | **YOLO-Based Wound Detection and Classification** | -- | 2023–2024 | Search on Google Scholar | Direct application of YOLO to wound detection |
| 2.4 | **Ultralytics YOLOv8: State-of-the-Art Real-Time Detection** | Jocher G. et al. | 2023 | GitHub: ultralytics/ultralytics | Official documentation of the YOLO family used in the project |
| 2.5 | **Instance Segmentation with YOLO: From YOLOv5 to YOLOv8** | -- | 2023 | Search on arXiv | Evolution of instance segmentation in the YOLO family |

---

<a id="3-unet-and-unetpp-architectures"></a>
## 3. U-Net & U-Net++ Architectures

> **Project relevance:** Stage 2 uses U-Net++ with an EfficientNet encoder for mask refinement within ROI crops.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 3.1 | **U-Net: Convolutional Networks for Biomedical Image Segmentation** | Ronneberger O. et al. | 2015 | `10.1007/978-3-319-24574-4_28` | Original U-Net paper — foundational for any medical segmentation work |
| 3.2 | **UNet++: A Nested U-Net Architecture for Medical Image Segmentation** | Zhou Z. et al. | 2018 | `10.1007/978-3-030-00889-5_1` | Original U-Net++ paper — the architecture used in this project |
| 3.3 | **UNet++: Redesigning Skip Connections to Exploit Multiscale Features** | Zhou Z. et al. | 2020 | `10.1109/TMI.2019.2959609` | Extended journal version with additional experiments |
| 3.4 | **Attention U-Net: Learning Where to Look for the Pancreas** | Oktay O. et al. | 2018 | `arXiv:1804.03999` | Attention mechanism with U-Net — useful for comparison |
| 3.5 | **TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation** | Chen J. et al. | 2021 | `arXiv:2102.04306` | Modern architecture combining Transformers with U-Net — for future comparison |

---

<a id="4-mask-rcnn-for-medical-imaging"></a>
## 4. Mask R-CNN for Medical Imaging

> **Project relevance:** The primary model in the `experiments/maskrcnn/` track — Mask R-CNN with ResNet-50-FPN backbone.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 4.1 | **Mask R-CNN** | He K. et al. | 2017 | `10.1109/ICCV.2017.322` | Original paper — one of the project's core models |
| 4.2 | **Feature Pyramid Networks for Object Detection** | Lin T.-Y. et al. | 2017 | `10.1109/CVPR.2017.106` | FPN backbone used in the project |
| 4.3 | **Mask R-CNN for Medical Image Segmentation: A Review** | -- | 2022 | Search on Google Scholar | Review of Mask R-CNN applications in medical imaging |
| 4.4 | **Deep Residual Learning for Image Recognition** | He K. et al. | 2016 | `10.1109/CVPR.2016.90` | ResNet-50 — the backbone architecture |
| 4.5 | **Instance Segmentation of Skin Lesions Using Mask R-CNN** | -- | 2021 | Search on Google Scholar | Mask R-CNN applied to skin lesion segmentation |

---

<a id="5-postoperative-wound-infection-detection"></a>
## 5. Postoperative Wound Infection Detection

> **Project relevance:** The ultimate goal — classifying infection indicators (edema, hyperemia, necrosis) and determining infection status.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 5.1 | **Automated Surgical Site Infection Detection Using Deep Learning** | -- | 2022–2024 | Search on PubMed | Automated SSI detection with deep learning |
| 5.2 | **Computer-Aided Diagnosis of Wound Infection from Clinical Images** | -- | 2023 | Search on Google Scholar | Computer-aided wound infection diagnosis |
| 5.3 | **Surgical Site Infection Prediction Using Machine Learning: A Systematic Review** | -- | 2023 | Search on PubMed | Systematic review of ML-based SSI prediction |
| 5.4 | **Deep Learning for Wound Assessment: Current Status and Future Directions** | -- | 2023 | Search on IEEE Xplore | Deep learning for wound assessment — state of the art |
| 5.5 | **Infection Indicators in Chronic Wounds: Clinical and Computational Perspectives** | -- | 2022 | Search on PubMed | Infection indicators from clinical and computational perspectives |
| 5.6 | **Classification of Wound Tissue Types Using Color and Texture Features** | -- | 2020 | Search on Google Scholar | Wound tissue classification — basis for the infection MLP classifier |

---

<a id="6-medical-image-augmentation"></a>
## 6. Data Augmentation in Medical Imaging

> **Project relevance:** Medical augmentation strategy (preserving 3x3 cm marker geometry), both online and offline augmentation.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 6.1 | **A Survey on Image Data Augmentation for Deep Learning** | Shorten C. & Khoshgoftaar T. | 2019 | `10.1186/s40537-019-0197-0` | Comprehensive survey on data augmentation techniques |
| 6.2 | **Data Augmentation for Medical Image Segmentation** | -- | 2022 | Search on Google Scholar | Augmentation techniques specific to medical image segmentation |
| 6.3 | **Albumentations: Fast and Flexible Image Augmentations** | Buslaev A. et al. | 2020 | `10.3390/info11020125` | The augmentation library used in this project |
| 6.4 | **Data Augmentation Strategies for Improving Wound Segmentation** | -- | 2023 | Search on Google Scholar | Augmentation strategies specifically for wound images |
| 6.5 | **Geometric-Preserving Augmentation for Medical Reference Markers** | -- | 2022 | Search on Google Scholar | Geometry-preserving augmentation — relevant to marker preservation |

---

<a id="7-transfer-learning-in-medical-imaging"></a>
## 7. Transfer Learning in Medical Imaging

> **Project relevance:** All models use ImageNet-pretrained weights as initialization.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 7.1 | **How Transferable Are Features in Deep Neural Networks?** | Yosinski J. et al. | 2014 | `arXiv:1411.1792` | Foundational work on feature transferability |
| 7.2 | **Transfer Learning for Medical Image Analysis: A Literature Review** | Kim H.E. et al. | 2022 | `10.1186/s12880-022-00793-7` | Comprehensive review of transfer learning in medical imaging |
| 7.3 | **Transfusion: Understanding Transfer Learning for Medical Imaging** | Raghu M. et al. | 2019 | `arXiv:1902.07208` | In-depth study on the effectiveness of ImageNet transfer to medical domains |
| 7.4 | **ImageNet-Trained CNNs Are Biased Towards Textures** | Geirhos R. et al. | 2019 | `arXiv:1811.12231` | Important for understanding what pretrained weights actually transfer |

---

<a id="8-cascaded-detection-segmentation-pipelines"></a>
## 8. Cascaded Detection–Segmentation Pipelines

> **Project relevance:** The core architecture YOLO -> U-Net++ -> MLP is a three-stage cascaded pipeline.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 8.1 | **Cascade R-CNN: Delving into High Quality Object Detection** | Cai Z. & Vasconcelos N. | 2018 | `10.1109/CVPR.2018.00644` | Foundational concept of cascaded pipelines in detection |
| 8.2 | **Two-Stage Detection-Then-Segmentation Approaches in Medical Imaging** | -- | 2023 | Search on Google Scholar | Detection-then-segmentation pipelines in medical imaging |
| 8.3 | **Hybrid Detection-Segmentation Models for Lesion Analysis** | -- | 2022 | Search on Google Scholar | Hybrid models combining detection and segmentation |
| 8.4 | **Multi-Stage Deep Learning Pipeline for Wound Analysis** | -- | 2023 | Search on PubMed | Multi-stage deep learning pipelines for wound analysis |

---

<a id="9-computerized-wound-area-measurement"></a>
## 9. Computerized Wound Area Measurement

> **Project relevance:** Computing wound area in cm² using the 3x3 cm reference marker.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 9.1 | **Smartphone-Based Wound Area Measurement Using Deep Learning** | -- | 2022 | Search on PubMed | Smartphone wound area measurement with deep learning |
| 9.2 | **Automated Wound Area Estimation Using Reference Markers and Segmentation** | -- | 2021 | Search on Google Scholar | Area estimation using reference markers |
| 9.3 | **Planimetric Wound Measurement: A Review of Methods** | -- | 2020 | Search on PubMed | Review of planimetric wound measurement methods |
| 9.4 | **Calibrated Wound Measurement Systems in Clinical Practice** | -- | 2022 | Search on PubMed | Calibrated measurement systems in clinical practice |

---

<a id="10-evaluation-metrics"></a>
## 10. Evaluation Metrics (COCO AP, Dice, IoU)

> **Project relevance:** The project uses combined COCO AP50, Dice, and IoU as primary evaluation metrics.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 10.1 | **Microsoft COCO: Common Objects in Context** | Lin T.-Y. et al. | 2014 | `10.1007/978-3-319-10602-1_48` | Defines the COCO AP metrics used in the project |
| 10.2 | **A Survey of Evaluation Metrics Used for Semantic & Instance Segmentation** | -- | 2023 | Search on Google Scholar | Comprehensive survey of evaluation metrics |
| 10.3 | **Metrics Reloaded: Recommendations for Image Analysis Validation** | Maier-Hein L. et al. | 2024 | `10.1038/s41592-023-02151-z` | Modern guidelines for choosing appropriate metrics |
| 10.4 | **V-Net: Fully Convolutional Neural Networks for Volumetric Segmentation** | Milletari F. et al. | 2016 | `10.1109/3DV.2016.79` | The paper that introduced Dice Loss |

---

<a id="11-efficientnet-and-efficient-architectures"></a>
## 11. EfficientNet & Efficient Architectures

> **Project relevance:** U-Net++ uses EfficientNet-B1 as its encoder backbone.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 11.1 | **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks** | Tan M. & Le Q.V. | 2019 | `arXiv:1905.11946` | Original paper for the encoder used in the project |
| 11.2 | **EfficientNetV2: Smaller Models and Faster Training** | Tan M. & Le Q.V. | 2021 | `arXiv:2104.00298` | Improved version — useful for comparison |
| 11.3 | **Segmentation Models Pytorch** | Yakubovskiy P. | 2019 | GitHub: qubvel/segmentation_models.pytorch | The library used to build U-Net++ |

---

<a id="12-focal-loss-and-dice-loss"></a>
## 12. Focal Loss & Dice Loss

> **Project relevance:** The combined Focal + Dice loss function used in U-Net++ training.

### Seed Papers

| # | Title | Authors | Year | DOI / Identifier | Why this paper? |
|---|-------|---------|------|------------------|-----------------|
| 12.1 | **Focal Loss for Dense Object Detection** | Lin T.-Y. et al. | 2017 | `10.1109/ICCV.2017.324` | Original Focal Loss paper |
| 12.2 | **The Lovász-Softmax Loss: A Tractable Surrogate for the Submodular Optimization of IoU** | Berman M. et al. | 2018 | `arXiv:1705.08790` | Alternative to Dice — useful for comparison |
| 12.3 | **Unified Focal Loss: Generalising Dice and Cross Entropy-Based Losses to Handle Class Imbalanced Medical Image Segmentation** | Yeung M. et al. | 2022 | `10.1016/j.media.2021.102026` | Unified framework combining Focal and Dice losses |

---

<a id="search-queries"></a>
## 13. Suggested Search Queries for ResearchRabbit

Use these queries to search in ResearchRabbit, Google Scholar, or Semantic Scholar:

### Core Queries

```
1.  "wound segmentation deep learning"
2.  "surgical site infection detection computer vision"
3.  "postoperative wound assessment machine learning"
4.  "YOLO medical image segmentation"
5.  "U-Net++ medical image segmentation"
6.  "Mask R-CNN wound detection"
7.  "wound area measurement reference marker"
8.  "cascaded detection segmentation pipeline medical imaging"
9.  "transfer learning medical image analysis"
10. "data augmentation medical image segmentation"
```

### Specialized Queries

```
11. "wound infection classification deep learning clinical images"
12. "EfficientNet encoder segmentation medical"
13. "Focal Loss Dice Loss combined segmentation"
14. "COCO evaluation metrics medical imaging"
15. "wound tissue classification granulation fibrin necrosis"
16. "smartphone wound imaging automated assessment"
17. "instance segmentation skin lesion clinical photograph"
18. "two-stage detection refinement segmentation"
19. "wound healing monitoring computer vision longitudinal"
20. "calibrated wound measurement planimetry deep learning"
```

### Clinical Context Queries

```
21. "surgical site infection criteria CDC guidelines"
22. "wound assessment scoring systems clinical"
23. "telemedicine wound care remote monitoring"
24. "chronic wound management artificial intelligence"
25. "wound bed preparation tissue classification clinical"
```

---

<a id="collection-building-plan"></a>
## 14. Collection Building Plan in ResearchRabbit

### Step 1: Create Collections

Create the following collections in ResearchRabbit:

| Collection | Seed Papers | Purpose |
|------------|-------------|---------|
| **Wound Segmentation Core** | 1.1, 1.2, 1.3, 1.4 | Theoretical foundation for wound segmentation |
| **YOLO + Detection** | 2.1–2.5, 4.1, 4.2 | Detection architectures used in the project |
| **U-Net Family** | 3.1–3.5 | Semantic segmentation architectures |
| **Infection Detection** | 5.1–5.6 | Infection detection — the primary objective |
| **Pipeline & Methods** | 6.1–6.5, 7.1–7.4, 8.1–8.4 | Techniques and methodologies |
| **Measurement & Metrics** | 9.1–9.4, 10.1–10.4 | Measurement and evaluation |
| **Architecture Components** | 11.1–11.3, 12.1–12.3 | Architecture components (encoder, loss functions) |

### Step 2: Explore Connections

After adding the seed papers:
1. Use **"Similar Work"** to discover related papers
2. Use **"All References"** to trace foundational references
3. Use **"Citations"** to find follow-up works
4. Focus on papers published between **2020–2026** for recency

### Step 3: Filter and Organize

- Sort by **citation count** to identify the most influential works
- Look for **survey/review papers** for comprehensive coverage
- Prioritize **peer-reviewed journals** (IEEE TMI, Medical Image Analysis, Nature Methods, etc.)

### Step 4: Map References to Thesis Chapters

| Thesis Chapter | Relevant Collections |
|----------------|----------------------|
| **Chapter 1: Introduction** | Wound Segmentation Core, Infection Detection |
| **Chapter 2: Theoretical Background** | U-Net Family, YOLO + Detection, Architecture Components |
| **Chapter 3: Related Work** | All collections |
| **Chapter 4: Methodology** | Pipeline & Methods, Measurement & Metrics |
| **Chapter 5: Experiments & Results** | Measurement & Metrics, all comparison-related papers |
| **Chapter 6: Discussion** | Infection Detection, Wound Segmentation Core |

---

## Important Notes

1. **Papers listed without DOI:** Search for the title on Google Scholar or Semantic Scholar,
   then add the paper to ResearchRabbit via its DOI or article link.

2. **Periodic updates:** Re-run searches every 2–3 months to discover newly published papers.

3. **Complementary tools:**
   - [Semantic Scholar](https://www.semanticscholar.org/) — for semantic search
   - [Connected Papers](https://www.connectedpapers.com/) — for visualizing paper networks
   - [Elicit](https://elicit.com/) — for extracting information from papers
   - [Zotero](https://www.zotero.org/) — for reference management

4. **Export:** You can export ResearchRabbit collections to Zotero or BibTeX
   for direct use in LaTeX or Word.

---

> **Last updated:** April 2026
> **Project:** Wound Infection Detection Model — Master's Thesis
