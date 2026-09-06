# VOICE PIPELINE SPECIFICATION FOR GOOGLE ANTIGRAVITY

## Project: Hierarchical Multimodal Emotion Recognition (MER) for Vietnamese Journaling Monologues

---

### 1\. OVERVIEW & MISSION FOR ANTIGRAVITY AGENT

You are an expert AI software engineer and Deep Learning architect inside Google Antigravity. Your mission is to scaffold, implement, and verify a complete, modular, production-ready PyTorch codebase for a **Speech-derived Bimodal Emotion Recognition (Voice MER) Pipeline** specifically tailored for Vietnamese expressive journaling.

The pipeline takes a single raw audio recording (16kHz mono, 1-3 minutes) and predicts:

1. **Primary Emotion (5 classes):** Joy/Positive, Sadness, Anxiety/Fear, Anger/Frustration, Neutral.  
2. **Sub-category Emotions (Fine-grained):** Gratitude, Pride, Relief, Disappointment, Remorse, Loneliness, Nervousness, Fear, Annoyance, Realization.

---

### 2\. MATHEMATICAL ARCHITECTURE & TENSOR SPECIFICATIONS

\[Raw Audio Input: (B, T), 16kHz\]

        │

        ├───────────────────────────────┬───────────────────────────────┐

        ▼                               ▼                               ▼  
 \[Stream 1: Acoustic\]          \[Stream 2: Semantic\]        \[Stream 3: Paralinguistic\]  
 \[Stream 1: Acoustic\]          \[Stream 2: Semantic\]        \[Stream 3: Paralinguistic\]

 Silero-VAD \+ Chunking         PhoWhisper-base             Pause & Hesitation Analysis

 (B, N, 6s\*16000)              Text Transcripts (B strings) P\_pause: (B, 4\)

        │                               │                               │

 emotion2vec-base \+ LoRA       PhoBERT-base Encoder        MLP Projector

 (r=8 on attention layers)     z\_semantic: (B, 768\)        Z\_pause: (B, 128\)

 z\_i: (B, N, 768\)                       │                               │

        │                               │                               │

 Temporal Attention Pooling             │                               │

 Z\_audio: (B, 768\)                      │                               │

        │                               │                               │

        └───────────────────────────────┼───────────────────────────────┘

                                        ▼

                        \[Cross-Modal Gated Fusion\]

              Z\_fused \= \[ g ⊙ Z\_audio \+ (1-g) ⊙ z\_semantic ; Z\_pause \]

                               (B, 896\)

                                        │

                    ┌───────────────────┴───────────────────┐

                    ▼                                       ▼

       \[Head 1: Primary 5-Class\]               \[Head 2: Fine-grained Sub-class\]

       Softmax \+ Focal Loss (γ=2.0)            Conditioned on z\_semantic

#### Tensor Shapes:

- **Audio Batch:** $X\_{audio} \\in \\mathbb{R}^{B \\times T}$  
- **Chunks:** $N$ chunks of 6.0 seconds ($6.0 \\times 16000 \= 96000$ samples) with 50% overlap (3.0s stride).  
- **Acoustic Embeddings:** $z\_i \\in \\mathbb{R}^{B \\times N \\times 768}$.  
- **Temporal Attention Weights:** $\\alpha \\in \\mathbb{R}^{B \\times N}, \\quad \\sum\_{i=1}^N \\alpha\_i \= 1$.  
- **Aggregated Audio Vector:** $Z\_{audio} \= \\sum\_{i=1}^N \\alpha\_i z\_i \\in \\mathbb{R}^{B \\times 768}$.  
- **Semantic Vector:** $Z\_{semantic} \\in \\mathbb{R}^{B \\times 768}$ từ `vinai/phobert-base`.  
- **Paralinguistic Vector:** $P\_{pause} \= \[\\text{SPR}, \\text{MeanDuration}, \\text{Frequency}, \\text{EnergyDrift}\] \\in \\mathbb{R}^{B \\times 4} \\to \\text{MLP} \\to Z\_{pause} \\in \\mathbb{R}^{B \\times 128}$.  
- **Gating Vector:** $g \= \\sigma(W\_g \[Z\_{audio} ,;, Z\_{semantic}\] \+ b\_g) \\in \\mathbb{R}^{B \\times 768}$.  
- **Fused Representation:** $Z\_{fused} \= \[g \\odot Z\_{audio} \+ (1 \- g) \\odot Z\_{semantic} ,;, Z\_{pause}\] \\in \\mathbb{R}^{B \\times 896}$.  
- **Primary Head Output:** $\\hat{y}\_{primary} \\in \\mathbb{R}^{B \\times 5}$.  
- **Sub-category Head Output:** $\\hat{y}\_{sub} \\in \\mathbb{R}^{B \\times 10}$.

---

### 3\. REPOSITORY DIRECTORY STRUCTURE TO SCAFFOLD

Generate the following modular codebase:  
voice\_mer\_project/  
├── configs/

│   ├── voice\_only.yaml

│   ├── text\_only.yaml

│   └── multimodal\_fusion.yaml

├── src/

│   ├── \_\_init\_\_.py

│   ├── audio/                  \# PHÂN HỆ ÂM HỌC (Voice-only)

│   │   ├── \_\_init\_\_.py

│   │   ├── vad\_preprocessor.py \# Silero-VAD \+ chunking \+ trích xuất chỉ số ngập ngừng

│   │   ├── emotion2vec\_lora.py \# emotion2vec-base gắn LoRA (r=8) thích ứng thanh điệu VN

│   │   └── attention\_pooling.py\# Temporal Attention Pooling trích xuất cao trào

│   ├── text/                   \# PHÂN HỆ VĂN BẢN (Text-only & Shared Text Modules)

│   │   ├── \_\_init\_\_.py

│   │   ├── text\_preprocessor.py\# Chuẩn hóa văn bản nhật ký tiếng Việt

│   │   └── text\_encoder.py     \# PhoBERT-base trích xuất ngữ nghĩa vi tế

│   ├── multimodal/             \# PHÂN HỆ HỢP NHẤT (Multimodal Fusion & ASR)

│   │   ├── \_\_init\_\_.py

│   │   ├── asr\_transcriber.py  \# PhoWhisper-base ASR chuyển âm thanh sang văn bản

│   │   ├── gated\_fusion.py     \# Cross-modal Gated Fusion layer (d=896)

│   │   └── hierarchical\_mer.py \# Mô hình tổng thể End-to-End với Dual Classification Heads

│   ├── data/                   \# QUẢN LÝ DỮ LIỆU

│   │   ├── \_\_init\_\_.py

│   │   └── dataset.py          \# PyTorch Dataset hỗ trợ Audio-only, Text-only và Audio+Text

│   ├── losses/                 \# HÀM MẤT MÁT

│   │   ├── \_\_init\_\_.py

│   │   └── focal\_loss.py       \# Multi-class Focal Loss (gamma=2.0)

│   └── utils/                  \# CÔNG CỤ ĐO LƯỜNG

│       ├── \_\_init\_\_.py  
│       └── metrics.py          \# Tính toán WA, UA, Macro-F1  
├── tests/  
│   ├── test\_voice\_pipeline.py  \# Test riêng nhánh Voice  
│   ├── test\_text\_pipeline.py   \# Test riêng nhánh Text  
│   └── test\_multimodal\_pass.py \# Test toàn bộ luồng Hợp nhất  
├── train\_voice.py              \# Script huấn luyện riêng Voice-only  
Lý do: PhoWhisper được tối ưu hóa chuyên biệt cho tiếng Việt với khả năng phục hồi dấu câu và viết hoa (punctuation & capitalization restoration), giúp giảm đáng kể Word Error Rate (WER) và đồng bộ trực tiếp với PhoBERT-base.  
├── train\_multimodal.py         \# Script huấn luyện mô hình Hợp nhất  
├── evaluate.py                 \# Đánh giá và xuất báo cáo kết quả  
└── requirements.txt  
---

### 6\. ACTION PLAN FOR ANTIGRAVITY

1. **Initialize Project:** Create folder structure and setup virtual environment with `requirements.txt`.  
2. Implement Modules: Create all .py files in src/audio/, src/text/, and src/multimodal/.  
3. Synthetic Dummy Test: Run tests/test\_multimodal\_pass.py using dummy audio tensors $(B=2, T=16000 \\times 15)$ to verify that gradients propagate cleanly through both Heads.  
4. Data Loader Implementation: Ensure dataset.py can load a folder of .wav files with a CSV containing \[file\_path, primary\_label, sub\_label\].  
5. Training Execution: Provide command line interface for python train\_multimodal.py \--config configs/multimodal\_fusion.yaml.

