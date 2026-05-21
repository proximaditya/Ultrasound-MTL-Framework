# Ultrasound-MTL-Framework
# Multi-Task Learning Framework for Abdominal Ultrasound Analysis

**Live Inference Module:** [Hugging Face Space](https://huggingface.co/spaces/ProximAditya/Ultrasound-AI)

## Abstract
This repository contains the implementation of a Multi-Task Learning (MTL) framework designed for the automated analysis of abdominal ultrasound images. Due to inherent speckle noise, low contrast, and operator dependency in ultrasound imaging, single-task convolutional networks often suffer from feature entanglement when simultaneously assessing structural anatomy and tissue pathology. 

To mitigate this, we propose a Dual-Head ConvNeXt architecture. A shared backbone extracts generalized morphological features, while two independent task heads decouple the specific classification objectives:
*   **Head 1 (Anatomical Classification):** Identifies 11 distinct organ classes.
*   **Head 2 (Pathological Detection):** Classifies the tissue state as normal or abnormal.

## Dataset and Preprocessing
The dataset exhibits severe class imbalance, skewed heavily towards normal patient statuses and specific organs (e.g., urinary bladder, prostate, and kidneys). Furthermore, data forensics identified extreme minority classes (underrepresented physiological states) which caused early gradient collapse. These specific anomalies were programmatically filtered prior to splitting.

[<img width="1384" height="583" alt="image" src="https://github.com/user-attachments/assets/6e65e940-846a-42e3-8fdd-90d5cb21b25e" />
]
*Figure 1: Distribution of normal vs. abnormal cases and detailed organ class frequencies.*

To address the low signal-to-noise ratio typical of raw ultrasound data, a preprocessing pipeline utilizing Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied. This equalizes pixel intensity distributions and enhances structural edges without amplifying background acoustic noise.

[<img width="1541" height="483" alt="image" src="https://github.com/user-attachments/assets/7b886bd1-0e9e-4fbc-82b3-3acc73c9146c" />
]
*Figure 2: Feature extraction and pixel intensity equalization on a normal liver sample.*

[<img width="1541" height="483" alt="image" src="https://github.com/user-attachments/assets/943528d7-1ea1-435d-92dd-c9ff4dd14c50" />
]
*Figure 3: Feature extraction and pixel intensity equalization on an abnormal ascites sample.*


## Technical Challenges & Architectural Evolution

The final Multi-Task Learning (MTL) framework is the culmination of iterative problem-solving to address several critical bottlenecks inherent to medical image analysis:

### 1. Data Forensics and Split Integrity
Initial directory diagnostics revealed severe statistical anomalies, notably extreme minority classes (e.g., the `abnormal_portal` class containing a singular sample). Passing these directly into standard splitting algorithms causes fatal stratified splitting errors. A dynamic pre-training filter was engineered to autonomously detect and drop statistically impossible classes ($n < 2$) prior to dataloader initialization, ensuring split integrity.

### 2. Feature Entanglement in Single-Task Baselines
The baseline architecture (EfficientNet-B0 at `224x224`) utilized dynamic class weighting to combat imbalance. While it achieved a 78% global accuracy, decoupled evaluation revealed a distinct bias: the model accurately classified macroscopic anatomy but failed to identify microscopic pathological textures. The single classification head suffered from feature entanglement, unable to mathematically optimize for structural boundaries and tissue textures simultaneously.

### 3. Gradient Instability and Numerically Safe Focal Loss
To capture pathological micro-textures, the input resolution was scaled to High-Definition (`384x384`) and the backbone was upgraded to ConvNeXt. However, applying extreme dynamic class weights to standard Focal Loss induced exploding gradients (loss spiking > 40.0), leading to immediate weight collapse. 

**Solution:** A custom, numerically stable Focal Loss was engineered. By decoupling the cross-entropy probability calculation from the focal modulation, and explicitly applying class weights strictly *after* probabilities were extracted, the gradient propagation was completely stabilized.

### 4. Domain-Specific Augmentation Strategy
Standard image augmentations are sub-optimal for sonography. The `Albumentations` pipeline was customized specifically for ultrasound physics:
*   **Affine Transformations:** Shift, scale, and rotate matrices were applied to simulate the physical pressure, angle, and movement of a sonographer's probe.
*   **Gaussian Noise & CLAHE:** Forced the network to become invariant to inherent acoustic speckle noise while enhancing low-contrast organ boundaries.



## Methodology

### 1. Network Architecture
*   **Backbone:** `ConvNeXt-Tiny`, configured for high-resolution input tensors (`384x384`). The modernized macro-design of ConvNeXt provides an optimal receptive field for capturing both macro-structures (organ boundaries) and micro-textures (pathological indicators).
*   **Multi-Task Heads:** The network diverges at the final classification layer. The structural head outputs logits for the 11 anatomical classes, while the diagnostic head outputs logits for binary pathology detection.

### 2. Loss Optimization
Due to the statistical imbalance across both the organ and pathology axes, standard Cross-Entropy Loss resulted in early gradient instability. This is addressed by implementing a mathematically stabilized, dynamically weighted Focal Loss function. The loss heavily penalizes the model for misclassifying minority pathology classes while ensuring gradients remain numerically safe during backpropagation.

## Experimental Results
The model was evaluated on a stratified holdout test set of **585 images**. The multi-task decoupling demonstrated robust generalization across both objectives, preventing the diagnostic head from overfitting to the structural head's features.

**Quantitative Performance:**
*   **Head 1 (Organ Classification Accuracy):** `90.77%`
*   **Head 2 (Disease Detection Accuracy):** `80.00%`

### Confusion Matrices
The decoupled evaluation highlights the network's high confidence in structural identification and its diagnostic sensitivity.

[<img width="1923" height="784" alt="image" src="https://github.com/user-attachments/assets/47a5701a-265a-4560-94d9-0586f6fa33d9" />
]
*Figure 4: Independent evaluation matrices for Organ Identification (left) and Disease Detection (right).*

For granular analysis of model predictions across specific condition-organ pairs during the single-head high-definition trials, the combined feature performance is detailed below:

[<img width="1470" height="1183" alt="image" src="https://github.com/user-attachments/assets/245ce639-2ce6-4386-89d5-56daa62fe604" />
]
*Figure 5: High-resolution confusion matrix detailing model predictions across combined physiological states.*

## Reproduction and Usage

### Prerequisites
*   Python 3.10+
*   PyTorch 2.0+

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/YourUsername/Ultrasound-MTL-Framework.git
cd Ultrasound-MTL-Framework
pip install -r requirements.txt
