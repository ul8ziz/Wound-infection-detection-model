# Technical Report: Postoperative Wound Infection Detection Using Deep Learning

## 1. Project Overview

### 1.1 Problem Definition

This project addresses the automated detection and analysis of postoperative wound infections from clinical photographs. The primary objective is to develop a computer vision system capable of:

- **Wound Detection and Segmentation**: Accurately identifying and segmenting postoperative wounds in clinical images
- **Area Measurement**: Computing wound area in square centimeters (cm²) using a reference marker
- **Infection Indicator Detection**: Identifying clinical signs of infection, including:
  - Edema (swelling) around the wound
  - Hyperemia (increased blood flow/redness)
  - Necrosis (tissue death)
  - Granulation tissue
  - Fibrin deposits
  - Suture zones

### 1.2 Medical Relevance and Motivation

Postoperative wound infection is a significant clinical concern that requires timely detection and monitoring. Traditional assessment methods rely on visual inspection by healthcare professionals, which can be subjective and time-consuming. An automated system that can:

1. **Quantify wound area** accurately using a standardized reference marker (3×3 cm square)
2. **Detect infection indicators** objectively from image features
3. **Provide structured output** in JSON format for clinical documentation

This system has the potential to support clinical decision-making, enable remote monitoring, and standardize wound assessment protocols. The project is developed as part of a Master's thesis in medical computer vision.

---

## 2. Dataset Description

### 2.1 Image Sources

The dataset consists of clinical photographs of postoperative wounds collected from approximately **240 separate CVAT annotation tasks** (task_0 through task_239). Each task contains:

- High-resolution clinical photographs (variable dimensions, typically JPEG/PNG format)
- Polygon-based annotations in COCO format
- Multi-class, multi-instance segmentation labels

### 2.2 Annotation Format

The dataset uses the **COCO (Common Objects in Context) annotation format**, which supports:

- **Polygon annotations** for precise segmentation masks
- **Bounding boxes** for object localization
- **Category labels** for multi-class classification
- **Image metadata** including dimensions and file paths

Annotations are organized in a hierarchical structure:
- Individual task annotations: `data/task_X/annotations.json`
- Merged annotations: `data/annotations.json` (consolidated from all tasks)
- Dataset splits: `data/splits/train.json`, `data/splits/val.json`, `data/splits/test.json`

### 2.3 Classes and Labels

The dataset includes **16 distinct classes** (plus background) representing different anatomical and pathological regions:

#### Primary Classes:
1. **"ВсяРана" (AllWound)** - Entire wound area (most frequent class)
2. **"Метка для размерности" (SizeMarker)** - 3×3 cm reference marker (critical for area conversion)

#### Infection Indicator Classes:
3. **"Зона отека вокруг раны" (EdemaZone)** - Swelling around wound (infection sign)
4. **"Зона гиперемии вокруг" (HyperemiaZone)** - Increased redness (infection sign)
5. **"Зона некроза" (NecrosisZone)** - Tissue necrosis (infection sign, rare but clinically important)

#### Tissue Type Classes:
6. **"Зона грануляций" (GranulationZone)** - Granulation tissue
7. **"Фибрин" (Fibrin)** - Fibrin deposits
8. **"Зона шва" (SutureZone)** - Suture area

#### Additional Classes:
- "Гнойное отделяемое" (Purulent discharge)
- "Сухожилие" (Tendon)
- "Губка ВАК" (VAC sponge)
- "Глубины раны" (Wound depths)
- And others

#### Class Distribution Characteristics:
- **Strong class imbalance**: "AllWound" and "SizeMarker" are most frequent
- **Rare classes**: Necrosis and other infection indicators appear less frequently
- **Background dominance**: Most of each image represents non-wound tissue

### 2.4 Infection Labeling Convention

Images are labeled for infection status using filename conventions:
- Images with **"-not-"** in the filename indicate **no infection**
- Images without this marker may contain infection indicators
- The model learns to distinguish texture and feature differences between infected and non-infected wounds

### 2.5 Data Augmentation Strategy

A comprehensive medical augmentation pipeline is implemented to address class imbalance and improve model generalization while preserving critical geometric properties.

#### Augmentation Pipeline Design

The augmentation strategy (`scripts/augmentation_strategy.py`) implements three intensity levels:

1. **Light**: Minimal transforms, maximum marker preservation
2. **Moderate**: Balanced augmentation (recommended default)
3. **Aggressive**: Strong augmentations (use with caution)

#### Key Augmentation Components

**Geometric Transforms (Marker-Preserving):**
- **Horizontal/Vertical Flips**: Preserve marker square geometry
- **Small Rotations**: Limited to ±10 degrees when marker preservation is enabled
- **Affine Transforms**: Controlled translation (±5%), scale (0.95-1.05), and minimal shear
- **Aspect Ratio Preservation**: Uses `LongestMaxSize` followed by padding to maintain marker proportions

**Photometric Transforms:**
- **Brightness/Contrast Adjustment**: Simulates varying clinical lighting conditions
- **Gaussian Noise**: Models sensor noise and image quality variations
- **Color Jitter**: Limited color space variations
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances local contrast

**Critical Constraints:**
- **Marker Geometry Preservation**: The 3×3 cm marker must remain approximately square to maintain accurate pixel-to-cm² conversion
- **Medical Realism**: Augmentations simulate realistic clinical variations without unrealistic distortions
- **Small Structure Preservation**: Rare classes (necrosis, granulation) require careful handling to avoid elimination

#### Offline vs. Online Augmentation

The system supports both approaches:

- **Offline Augmentation**: `scripts/apply_augmentation_only.py` generates augmented images and annotations saved to `data/augmented/` (typically 3 augmented versions per original image)
- **Online Augmentation**: Applied during training via `get_medical_augmentation_pipeline()` with configurable intensity

---

## 3. Model Architecture

### 3.1 Architecture Selection: Mask R-CNN

The project employs **Mask R-CNN (Region-based Convolutional Neural Network)** with a **ResNet-50 Feature Pyramid Network (FPN)** backbone. This architecture is selected for several reasons:

1. **Instance Segmentation Capability**: Mask R-CNN provides both bounding box detection and pixel-level segmentation masks, essential for accurate wound area calculation
2. **Multi-Class Detection**: Supports simultaneous detection of multiple classes (wound, marker, infection indicators) in a single forward pass
3. **Proven Performance**: Mask R-CNN is a well-established architecture with strong performance on medical segmentation tasks
4. **Feature Pyramid Network**: FPN backbone enables effective detection at multiple scales, important for wounds of varying sizes

### 3.2 Architecture Details

#### Backbone: ResNet-50-FPN

- **Base Network**: ResNet-50 (pretrained on ImageNet) provides feature extraction
- **Feature Pyramid Network**: Constructs multi-scale feature maps (P2-P6) for robust detection across scales
- **Pretrained Weights**: Uses "DEFAULT" weights from torchvision (pretrained on COCO dataset) for transfer learning

#### Detection Head

The model includes two prediction heads:

1. **Box Predictor (FastRCNNPredictor)**:
   - Input: Features from ROI Align (typically 1024 dimensions)
   - Output: Class scores and bounding box regression for `num_classes` classes
   - Architecture: Fully connected layers for classification and box regression

2. **Mask Predictor (MaskRCNNPredictor)**:
   - Input: Mask features from ROI Align (typically 256 channels)
   - Hidden Layer: 256-dimensional convolutional layer
   - Output: Binary segmentation masks for each detected instance
   - Architecture: Convolutional layers producing 28×28 masks (upsampled to original resolution)

#### Model Configuration

- **Number of Classes**: Automatically determined from dataset categories (typically 16 classes + background = 17 total)
- **Hidden Layer Size**: 256 (mask predictor)
- **Input Image Size**: 512×512 pixels (configurable, default for training)

### 3.3 Why Mask R-CNN for This Task

1. **Precise Segmentation**: Pixel-level masks enable accurate wound area calculation from segmentation masks rather than bounding boxes
2. **Multi-Instance Handling**: Can detect and segment multiple wound regions, markers, and infection indicators in a single image
3. **Class-Specific Masks**: Each detected instance has its own mask, allowing separate analysis of different tissue types
4. **Robust to Scale Variation**: FPN backbone handles wounds of different sizes effectively
5. **Transfer Learning**: Pretrained weights accelerate training and improve generalization

---

## 4. Training Methodology

### 4.1 Preprocessing Steps

#### Image Preprocessing

1. **Resizing**: Images are resized to a fixed size (default 512×512) while preserving aspect ratio using `LongestMaxSize` followed by padding
2. **Normalization**: Images are normalized using ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
3. **Tensor Conversion**: Images are converted to PyTorch tensors with channel-first format [C, H, W]

#### Annotation Preprocessing

1. **COCO Format Parsing**: Annotations are loaded using pycocotools COCO API or direct JSON parsing
2. **Polygon to Mask Conversion**: Polygon annotations are converted to binary segmentation masks
3. **Bounding Box Extraction**: Bounding boxes are computed from segmentation masks or extracted directly from annotations
4. **Target Dictionary Construction**: Each image is associated with a target dictionary containing:
   - `boxes`: Tensor of bounding boxes [N, 4] in [x1, y1, x2, y2] format
   - `labels`: Tensor of class IDs [N]
   - `masks`: Tensor of binary masks [N, H, W]
   - `image_id`: Unique image identifier
   - `area`: Area of each instance in pixels
   - `iscrowd`: Crowd flag (typically 0 for this dataset)

### 4.2 Training Configuration

#### Hyperparameters

The training configuration (defined in `CONFIG` dictionary) includes:

- **Epochs**: 50 (configurable)
- **Batch Size**: 4 (optimized for GPU memory constraints)
- **Learning Rate**: 0.005 (initial learning rate)
- **Image Size**: (512, 512) pixels
- **Random Seed**: 42 (for reproducibility)
- **Number of Workers**: 0 (Windows compatibility, can be increased on Linux/Mac)

#### Optimizer

- **Type**: Stochastic Gradient Descent (SGD)
- **Momentum**: 0.9
- **Weight Decay**: 0.0005 (L2 regularization)
- **Learning Rate**: 0.005 (initial)

#### Learning Rate Scheduler

- **Type**: StepLR (Step Learning Rate Scheduler)
- **Step Size**: 5 epochs
- **Gamma**: 0.1 (reduces learning rate by 10× every 5 epochs)
- **Scheduling**: Applied per-epoch (not per-iteration)

#### Loss Functions

Mask R-CNN uses a multi-task loss combining:

1. **Classification Loss**: Cross-entropy loss for object class prediction
2. **Bounding Box Regression Loss**: Smooth L1 loss for box coordinates
3. **Mask Segmentation Loss**: Binary cross-entropy loss for mask prediction
4. **RPN Losses**: Region Proposal Network losses for object proposal generation

The total loss is the sum of all component losses, automatically computed by the PyTorch Mask R-CNN implementation.

#### Training Procedure

1. **Model Initialization**: 
   - Load pretrained ResNet-50-FPN weights
   - Replace box and mask predictors with custom heads for the number of classes

2. **Training Loop**:
   - For each epoch:
     - Set model to training mode
     - Iterate through training batches
     - Forward pass: Compute predictions and losses
     - Backward pass: Compute gradients
     - Gradient clipping: Apply max_norm=1.0 to prevent exploding gradients
     - Optimizer step: Update model parameters
     - Log losses and metrics

3. **Validation**:
   - Set model to evaluation mode
   - Compute validation loss (using training mode with no_grad for loss computation)
   - Evaluate metrics using COCO evaluation API

4. **Checkpointing**:
   - Save `last.pt` after each epoch
   - Save `best.pt` when validation metric (combined_AP50) improves
   - Checkpoints include: model state, optimizer state, scheduler state, epoch number, best metric

### 4.3 Hardware Assumptions

The code is designed to work with:
- **GPU**: CUDA-capable GPU recommended (training on CPU is possible but significantly slower)
- **Memory**: Batch size of 4 with 512×512 images requires approximately 8-12 GB GPU memory
- **Device Selection**: Automatic (CUDA if available, otherwise CPU)

Training time estimates:
- **GPU (RTX 3090 or similar)**: 4-6 hours for 50 epochs
- **CPU**: Significantly longer (not recommended for full training)

---

## 5. Inference and Output

### 5.1 Prediction Generation

The inference pipeline consists of two main functions:

#### `run_inference()`

Performs basic inference on a single image:

1. **Image Loading**: Supports multiple input formats:
   - File path (string)
   - NumPy array [H, W, C] in RGB format
   - PyTorch tensor [C, H, W]

2. **Preprocessing**:
   - Resize to 512×512
   - Normalize using ImageNet statistics
   - Convert to tensor and add batch dimension

3. **Model Forward Pass**:
   - Model set to evaluation mode
   - Forward pass through Mask R-CNN
   - Extract predictions: boxes, scores, labels, masks

4. **Confidence Filtering**:
   - Filter predictions by confidence threshold (default: 0.5)
   - Return both filtered and raw predictions

#### `run_wound_inference()`

Specialized function for wound analysis:

1. **Inference Execution**: Calls `run_inference()` with configurable confidence threshold (default: 0.3 for higher recall)

2. **Marker Detection**:
   - Identifies reference marker by class ID
   - Selects marker with highest confidence if multiple detected
   - Computes marker area from segmentation mask (or bounding box as fallback)

3. **Pixel-to-cm² Conversion**:
   - Calculates conversion ratio: `pixel_to_cm2_ratio = marker_size_cm2 / marker_area_pixels`
   - Default marker size: 9.0 cm² (3×3 cm square)

4. **Wound Area Calculation**:
   - Identifies wound regions by class IDs
   - Sums pixel areas from segmentation masks
   - Converts to cm²: `wound_area_cm2 = wound_area_pixels × pixel_to_cm2_ratio`
   - Returns `None` if marker not found

5. **Infection Indicator Detection**:
   - Checks for presence of infection indicator classes (edema, hyperemia, necrosis)
   - Records detection status, maximum confidence score, and count for each indicator

### 5.2 JSON Output Structure

The system produces structured JSON output containing:

```json
{
  "wound_area_cm2": 25.3,
  "wound_area_pixels": 125000.0,
  "wound_area_ratio": 0.4768,
  "marker_found": true,
  "marker_area_pixels": 4500.0,
  "pixel_to_cm2_ratio": 0.002,
  "infection_flags": {
    "3": {
      "detected": true,
      "max_score": 0.87,
      "count": 2
    },
    "4": {
      "detected": true,
      "max_score": 0.75,
      "count": 1
    },
    "5": {
      "detected": false,
      "max_score": 0.0,
      "count": 0
    }
  },
  "num_detections": 5,
  "detections": [
    {
      "class_id": 1,
      "score": 0.92,
      "box": [100, 150, 300, 400],
      "has_mask": true
    }
  ],
  "raw_stats": {
    "num_raw": 8,
    "num_filtered": 5,
    "conf_thresh": 0.3
  }
}
```

#### Output Fields Explanation

- **`wound_area_cm2`**: Wound area in square centimeters (null if marker not found)
- **`wound_area_pixels`**: Wound area in pixels
- **`wound_area_ratio`**: Wound area as fraction of total image area
- **`marker_found`**: Boolean indicating if reference marker was detected
- **`marker_area_pixels`**: Marker area in pixels
- **`pixel_to_cm2_ratio`**: Conversion factor from pixels to cm²
- **`infection_flags`**: Dictionary mapping class IDs to detection information
- **`num_detections`**: Total number of detections after confidence filtering
- **`detections`**: List of all detections with class ID, score, bounding box, and mask availability

### 5.3 Post-Processing Steps

1. **Confidence Thresholding**: Filter low-confidence predictions (configurable threshold, typically 0.3-0.5)

2. **Non-Maximum Suppression (NMS)**: Handled internally by Mask R-CNN to remove duplicate detections

3. **Mask Thresholding**: Binary masks are thresholded at 0.5 to convert from probability maps to binary segmentation

4. **Area Calculation**: Pixel counts from binary masks are used for area computation (more accurate than bounding box area)

5. **Marker Validation**: System checks for marker presence before computing cm² measurements

---

## 6. Evaluation Strategy

### 6.1 Metrics Used

The evaluation system employs **COCO evaluation metrics** when pycocotools is available, with fallback to custom metrics:

#### COCO Metrics (Primary)

1. **Bounding Box Metrics**:
   - **bbox_AP**: Average Precision (AP) over all IoU thresholds (0.50:0.95)
   - **bbox_AP50**: AP at IoU threshold 0.50
   - **bbox_AP75**: AP at IoU threshold 0.75

2. **Segmentation Metrics**:
   - **segm_AP**: Average Precision for segmentation masks (IoU 0.50:0.95)
   - **segm_AP50**: Segmentation AP at IoU 0.50
   - **segm_AP75**: Segmentation AP at IoU 0.75

3. **Combined Metric**:
   - **combined_AP50**: Average of bbox_AP50 and segm_AP50 (used for model selection)

#### Fallback Metrics

When COCO evaluation is unavailable, the system uses custom metrics:

- **Precision**: True positives / (True positives + False positives) at IoU threshold 0.5
- **Recall**: True positives / (True positives + False negatives) at IoU threshold 0.5
- **F1 Score**: Harmonic mean of precision and recall
- **bbox_AP50**: Set equal to F1 score as proxy

### 6.2 Evaluation Procedure

1. **Model Evaluation Mode**: Model set to `eval()` mode (disables dropout, batch normalization uses running statistics)

2. **Prediction Generation**: 
   - Forward pass through validation dataset
   - Extract predictions: boxes, scores, labels, masks
   - Convert masks to binary (threshold 0.5) and RLE format for COCO evaluation

3. **COCO Evaluation**:
   - Load ground truth annotations using COCO API
   - Convert predictions to COCO result format
   - Run COCOeval for both bounding box and segmentation tasks
   - Extract AP statistics

4. **Metric Logging**: Metrics are logged per epoch and stored in training results JSON

### 6.3 Visual Validation

The system includes visualization capabilities in the training pipeline notebook:

1. **Prediction Visualization**:
   - Side-by-side comparison of ground truth and predictions
   - Color-coded bounding boxes by confidence level:
     - Red: High confidence (≥ 0.7)
     - Orange: Medium confidence (0.5 - 0.7)
     - Yellow: Low confidence (threshold - 0.5)
   - Ground truth boxes shown in green

2. **Statistics Display**:
   - Raw vs. filtered detection counts
   - Maximum confidence scores
   - Detection counts at different thresholds (0.3, 0.5, 0.7)

3. **Wound Analysis Visualization**:
   - Wound area display (pixels and cm²)
   - Marker detection status
   - Infection indicator flags
   - All detections with scores

### 6.4 Training Diagnostics

The validation function (`validate_one_epoch()`) includes optional prediction tracking:

- **Score Statistics**: Mean, median, max, min prediction scores
- **Threshold Counts**: Number of predictions above thresholds 0.3, 0.5, 0.7
- **Loss Components**: Individual loss components (classification, box regression, mask segmentation)

These diagnostics help identify training issues such as:
- Low recall (few predictions generated)
- Overconfidence (all scores very high)
- Underconfidence (all scores very low)

---

## 7. Strengths and Limitations

### 7.1 Strengths of the Approach

1. **Comprehensive Architecture**:
   - Mask R-CNN provides both detection and segmentation in a unified framework
   - FPN backbone enables robust multi-scale detection
   - Pretrained weights accelerate training and improve generalization

2. **Medical-Specific Design**:
   - Augmentation pipeline preserves critical marker geometry
   - Supports both offline and online augmentation strategies
   - Handles class imbalance through augmentation and careful transform selection

3. **Accurate Area Measurement**:
   - Uses segmentation masks rather than bounding boxes for area calculation
   - Reference marker enables pixel-to-cm² conversion
   - Handles cases where marker is not detected gracefully

4. **Structured Output**:
   - JSON format suitable for clinical documentation
   - Includes confidence scores for transparency
   - Provides both quantitative (area) and qualitative (infection flags) information

5. **Modular Codebase**:
   - Separation of concerns (data loading, training, inference)
   - Reusable components (augmentation, dataset, model building)
   - Support for both script-based and notebook-based workflows

6. **Evaluation Framework**:
   - COCO metrics provide standardized evaluation
   - Fallback metrics ensure evaluation works even without pycocotools
   - Per-epoch metric tracking enables training monitoring

### 7.2 Known Limitations

1. **Marker Dependency**:
   - Wound area in cm² requires marker detection
   - If marker is not detected, only pixel area is available
   - Marker must be visible and correctly annotated in training data

2. **Class Imbalance**:
   - Rare classes (necrosis, purulent discharge) may have limited training examples
   - Model may struggle with rare infection indicators
   - Augmentation helps but may not fully compensate for extreme imbalance

3. **Fixed Image Size**:
   - Images are resized to 512×512, which may lose detail in high-resolution clinical photos
   - Aspect ratio preservation with padding may introduce black borders
   - Larger image sizes would improve detail but increase computational cost

4. **Confidence Threshold Sensitivity**:
   - Model performance depends on appropriate confidence threshold selection
   - Low thresholds increase recall but may introduce false positives
   - High thresholds improve precision but may miss true positives
   - Optimal threshold may vary by class

5. **Limited Validation on External Data**:
   - Model trained and evaluated on single dataset
   - Generalization to different imaging conditions (lighting, camera types, patient populations) not extensively validated
   - Domain shift may affect performance in clinical deployment

6. **Computational Requirements**:
   - Training requires GPU for practical use (CPU training is extremely slow)
   - Inference is faster but still benefits from GPU acceleration
   - Batch size limited by GPU memory (typically 4 for 512×512 images)

7. **Annotation Quality Dependency**:
   - Model performance depends on annotation quality and consistency
   - Polygon annotations must accurately represent wound boundaries
   - Inconsistent marker placement or size may affect area calculations

8. **Binary Infection Classification**:
   - Current system detects infection indicators but does not provide a unified infection probability score
   - Infection determination requires interpretation of multiple indicator flags
   - No explicit model for distinguishing infected vs. non-infected wounds beyond indicator detection

---

## 8. Conclusion

### 8.1 Methodology Summary

This project implements a comprehensive deep learning system for postoperative wound infection detection using Mask R-CNN with ResNet-50-FPN backbone. The methodology encompasses:

1. **Data Management**: COCO-format annotations from 240 CVAT tasks, supporting multi-class, multi-instance segmentation
2. **Medical Augmentation**: Specialized augmentation pipeline preserving marker geometry while addressing class imbalance
3. **Instance Segmentation**: Mask R-CNN architecture providing both detection and pixel-level segmentation
4. **Quantitative Analysis**: Wound area calculation in cm² using reference marker-based pixel conversion
5. **Infection Detection**: Multi-class detection of clinical infection indicators (edema, hyperemia, necrosis, etc.)
6. **Structured Output**: JSON format suitable for clinical documentation and integration

### 8.2 Suitability for Medical Imaging Applications

#### Advantages for Medical Use

1. **Quantitative Measurements**: Provides objective wound area measurements in standard units (cm²), reducing subjectivity in clinical assessment

2. **Multi-Task Capability**: Simultaneously detects wound boundaries, reference markers, and infection indicators in a single forward pass

3. **Interpretability**: Confidence scores and structured output enable clinicians to assess model reliability

4. **Scalability**: Can process images automatically, potentially supporting remote monitoring and telemedicine applications

5. **Standardization**: Consistent measurement methodology could standardize wound assessment protocols across different healthcare settings

#### Considerations for Clinical Deployment

1. **Validation Requirements**: Extensive validation on diverse patient populations, imaging conditions, and wound types would be necessary before clinical deployment

2. **Regulatory Compliance**: Medical device regulations (e.g., FDA, CE marking) would require rigorous validation, documentation, and quality assurance

3. **Integration**: System would need integration with electronic health records (EHR) and clinical workflows

4. **Human Oversight**: Model predictions should be reviewed by healthcare professionals; the system is intended to assist, not replace, clinical judgment

5. **Ethical Considerations**: Patient privacy, data security, and informed consent for image use must be addressed

### 8.3 Research Contributions

This project demonstrates:

- **Medical-Specific Augmentation**: Development of augmentation strategies that preserve critical geometric properties (marker shape) while improving model generalization
- **Multi-Class Medical Segmentation**: Application of instance segmentation to complex medical scenes with multiple tissue types and infection indicators
- **Quantitative Wound Analysis**: Integration of reference marker-based area calculation into deep learning pipeline
- **End-to-End Pipeline**: Complete system from data preparation through training to inference and structured output

### 8.4 Future Directions

Potential improvements and extensions:

1. **Advanced Architectures**: Exploration of newer architectures (e.g., DETR, YOLO variants) for improved speed and accuracy
2. **Attention Mechanisms**: Integration of attention modules to focus on clinically relevant regions
3. **Multi-Task Learning**: Joint training for wound segmentation and infection classification
4. **Temporal Analysis**: Extension to video sequences for wound healing monitoring over time
5. **Uncertainty Quantification**: Integration of uncertainty estimation methods for more reliable predictions
6. **Federated Learning**: Training across multiple institutions while preserving patient privacy

---

## References

### Technical Documentation

- **Repository**: https://github.com/ul8ziz/Wound-infection-detection-model
- **Framework**: PyTorch 2.0+, torchvision
- **Architecture**: Mask R-CNN (He et al., 2017)
- **Backbone**: ResNet-50 with Feature Pyramid Network (Lin et al., 2017)
- **Augmentation**: Albumentations library
- **Evaluation**: COCO evaluation API (pycocotools)

### Key Files

- `notebooks/train_model.py`: Unified training and inference script
- `notebooks/pipeline_utils.py`: Data loading and preprocessing utilities
- `scripts/augmentation_strategy.py`: Medical augmentation pipeline
- `docs/DATA_AUGMENTATION_GUIDE.md`: Comprehensive augmentation documentation
- `notebooks/INFERENCE_GUIDE.md`: Inference and analysis guide

---

**Note**: This is a research project developed as part of a Master's thesis. The system is not intended for direct clinical use without appropriate validation, regulatory approval, and integration into clinical workflows. All predictions should be reviewed by qualified healthcare professionals.
