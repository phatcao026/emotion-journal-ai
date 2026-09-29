# TÀI LIỆU BÀN GIAO TOÀN DIỆN: TRIỂN KHAI GIAI ĐOẠN 2 (MULTIMODAL FUSION & RESEARCH ROADMAP)

* **Dự án:** Hệ thống Nhật ký Tâm lý & Nhận diện Cảm xúc Đa phương thức (Emotion Journal AI)
* **Workspace:** `d:\TravisL\emotion-journal-ai`
* **Nhánh Git hiện tại:** `feature/unify-structure`
* **Ngày cập nhật:** 2026-09-29

---

## 1. TÓM TẮT THÀNH TỰU STAGE 1.5 (ACOUSTIC BACKBONE)

### 1.1. Kiến trúc Mô hình Âm học Đã Hoàn Thành
* **Mô hình nền tảng:** `emotion2vec+` (iic/emotion2vec_base) kết hợp **LoRA Fine-tuning** ($r=8, \alpha=16$) trên các lớp attention projection.
* **Cơ chế gộp động:** **Temporal Attention Pooling** tự động tính trọng số khung biểu cảm ($\alpha_t$) thay cho Mean Pooling cơ học.
* **Phân tích khoảng lặng lâm sàng:** Module `EmotionalPauseAnalyzer` trích xuất vector 4 chiều $P_{\text{pause}} = [\text{SPR}, \text{MeanPauseDuration}, \text{PauseFrequency}, \text{EnergyDrift}] \in \mathbb{R}^4 \rightarrow$ chiếu phi tuyến lên $\mathbb{R}^{128}$.
* **Trọng số mô hình tốt nhất:** File checkpoint `best_model_stage1_5.pt` (374MB) đã được tải về máy cục bộ an toàn.

### 1.2. Dữ liệu Huấn luyện & Kết quả Thực nghiệm trên Tập Test Độc lập (632 mẫu mù)
* **Tổng dữ liệu chuẩn hóa 16kHz Mono:** 6,264 mẫu người thật (ViSEC 5,280 mẫu + RAVDESS 192 mẫu + CREMA-D 792 mẫu).
* **Phân bổ:** Train 5,009 mẫu (80%) | Val 623 mẫu (10%) | Test 632 mẫu (10%) - Stratified Split độc lập người nói.
* **Chỉ số đánh giá tổng thể:**
  * **Macro-F1:** `86.43%`
  * **Weighted Accuracy (WA):** `86.08%`
  * **Unweighted Accuracy (UA):** `86.97%`
* **Hiệu năng từng phân lớp:**
  * `ANXIETY` (Lo âu): Recall **100.0%**, Precision **95.2%**, **F1 = 0.975** (99/99 mẫu test) — triệt tiêu hiện tượng âm tính giả (False Negative = 0).
  * `ANGER` (Tức giận): Recall **93.9%**, F1 = **0.900** (148 mẫu).
  * `SADNESS` (Buồn bã): Recall **91.7%**, F1 = **0.855** (108 mẫu).
  * `JOY` (Vui vẻ): Recall **73.4%**, F1 = **0.800** (124 mẫu).
  * `NEUTRAL` (Bình ổn): Recall **75.5%**, F1 = **0.792** (153 mẫu).

---

## 2. KẾT LUẬN & CHỈ ĐẠO MỚI TỪ BUỔI HỌP VỚI GIẢNG VIÊN HƯỚNG DẪN (CÔ)

Từ biên bản họp với Cô, nhóm đã xác định 5 yêu cầu học thuật và định hướng kỹ thuật bắt buộc phải tích hợp vào Stage 2:

### 2.1. Đánh giá Thực nghiệm & "Hoán đổi bộ dữ liệu"
* Không chỉ dừng lại ở một lần chia tách 80/10/10 cố định.
* Cần triển khai:
  1. **5-Fold Stratified Cross-Validation:** Chia 5 phần, hoán đổi xoay vòng để báo cáo độ lệch chuẩn $(\mu \pm \sigma)$.
  2. **Leave-One-Corpus-Out (LOCO) Evaluation:** Huấn luyện trên 2 tập (RAVDESS + CREMA-D), kiểm thử zero-shot trực tiếp trên tập tiếng Việt ViSEC để đo lường định lượng khả năng chuyển giao âm học xuyên ngôn ngữ.

### 2.2. Trích xuất Đặc trưng Giai điệu Tường minh ("Melody" Features)
* **Hiện trạng:** Pipeline đã có *Melody dạng ẩn (Implicit Prosody)* nén trong vector 768 chiều của `emotion2vec+` và 4 chỉ số khoảng lặng trong `EmotionalPauseAnalyzer`.
* **Yêu cầu bổ sung của Cô:** Cần trích xuất thêm các đặc trưng giai điệu tường minh (Explicit Acoustic Biomarkers) có thể vẽ đồ thị đối chứng:
  * Đường cong cao độ $F_0$ (Pitch Contour qua pYIN / Kaldi Pitch).
  * Các thông số thống kê giai điệu: $F_0$ mean, $F_0$ std, $F_0$ range, độ dốc giai điệu (Pitch slope).
  * Chỉ số rung thanh quản: Jitter, Shimmer, HNR (Harmonics-to-Noise Ratio).
  * Vector kết hợp: $Z_{\text{acoustic\_final}} = [Z_{\text{emotion2vec}} (768) \,\|\, Z_{\text{pause}} (128) \,\|\, Z_{\text{melody\_explicit}} (32)]$.

### 2.3. Tách biệt Ngữ nghĩa (Content) và Ngữ điệu (Melody / Prosody)
* Cùng một câu nói (ví dụ: *"Tôi ổn lắm"*), nhưng nói với giai điệu khác nhau thì cảm xúc hoàn toàn khác nhau.
* Hệ thống phân tách thành 2 luồng:
  * **Luồng Ngữ nghĩa:** Voice $\rightarrow$ ASR (PhoWhisper-base) $\rightarrow$ Text Transcript $\rightarrow$ PhoBERT $\rightarrow Z_{\text{semantic}} \in \mathbb{R}^{768}$.
  * **Luồng Giai điệu/Âm sắc:** Voice $\rightarrow$ emotion2vec+ LoRA + Pause + Explicit Melody $\rightarrow Z_{\text{acoustic}} \in \mathbb{R}^{768+128+32}$.
* **Cơ chế Gated Fusion Unit:** Tự động điều chỉnh trọng số $g = \sigma(W [Z_{\text{acoustic}} \,\|\, Z_{\text{semantic}}])$ để bắt trọn các ca mỉa mai hoặc che giấu cảm xúc (nói câu tích cực nhưng giọng run rẩy, u uất).

### 2.4. Kế hoạch Thu thập 5,000 – 10,000 Dữ liệu Tiếng Việt Tự thân
* Xây dựng một ứng dụng nhỏ (Web Applet bằng Streamlit / FastAPI) để thu thập giọng nói cảm xúc người Việt đa vùng miền (Bắc, Trung, Nam).
* Gán nhãn dựa trên cơ sở sinh học và sàng lọc y tế chuẩn hóa: Thang đo trầm cảm **PHQ-9**, thang đo lo âu **GAD-7**, và mô hình chiều cảm xúc **VAD (Valence - Arousal - Dominance)**.

### 2.5. Khung sườn Bài báo Khoa học (Scientific Paper Structure)
* Cấu trúc 5 phần chính:
  1. *Data Collection:* Quy trình thu thập dữ liệu tự thân + Cross-corpus data.
  2. *Dataset Profiling:* Phân bố nhân khẩu học, cân bằng nhãn, thời lượng âm thanh.
  3. *Annotation Protocol:* Quy trình gán nhãn, độ đồng thuận Cohen's/Fleiss' Kappa.
  4. *Classification Architecture:* Pipeline phân tách Melody vs. Content, Gated Fusion, Hierarchical Dual-Head (5 lớp chính vs. 11 lớp con).
  5. *Biological & Psychological Grounding:* Cơ chế sinh học thần kinh thanh quản và chỉ dấu ức chế tâm vận động (Psychomotor Retardation).

---

## 3. TRẠNG THÁI HIỆN TẠI CỦA WORKSPACE & QUY TẮC BẮT BUỘC

### 3.1. Cấu trúc Thư mục Mã nguồn Đã Tái Cấu Trúc
```text
emotion-journal-ai/
├── configs/
│   ├── audio/           # voice_config.yaml, voice_only.yaml
│   ├── text/            # text_only.yaml
│   └── fusion/          # multimodal_fusion.yaml
├── data/
│   ├── taxonomy.json    # Ánh xạ chuẩn 5 Primary <-> 11 Sub-emotions
│   └── manifests/       # Metadata CSV
├── docs/
│   ├── business/        # SRS_v1.md, BAO_CAO_TIEN_DO_HOP_CO.md
│   ├── research/        # STAGE2_ACTION_PLAN_AND_SOLUTIONS.md
│   └── STAGE2_HANDOFF.md # File bàn giao này
├── src/
│   ├── common/          # metrics.py (WA, UA, Macro-F1), focal_loss.py
│   ├── audio/           # emotion2vec_lora, attention_pooling, pause_analyzer, asr_transcriber, train_voice, evaluate
│   ├── text/            # text_encoder (PhoBERT), text_preprocessor
│   └── fusion/          # gated_fusion, hierarchical_mer, train_multimodal
└── tests/               # 5 test suites tự động (audio, crosscorpus, multimodal, text, voice)
```

### 3.2. Trạng thái Git Cục bộ
* **Branch:** `feature/unify-structure`
* **3 file đang uncommitted trong working tree:**
  * `src/audio/evaluate.py`
  * `src/audio/train_voice.py`
  * `src/fusion/train_multimodal.py`
  *(Chứa bản vá `sys.path.insert(0, str(ROOT_DIR))` để hỗ trợ chạy trực tiếp CLI không cần cờ `-m`. Chủ ý giữ lại để gộp (squash) vào commit đầu tiên của Stage 2)*.
* **Quy tắc tuyệt đối:** Không tự ý chạy `git commit` hoặc `git push` khi chưa giải thích và được người dùng đồng ý tường minh.

---

## 4. KẾ HOẠCH HÀNH ĐỘNG KỸ THUẬT NGAY TRONG STAGE 2

1. **Xây dựng Module `src/audio/melody_extractor.py`:**
   * Trích xuất đường cao độ $F_0$ (Pitch Contour), $F_0$ statistics (mean, std, range, slope), Jitter, Shimmer, HNR.
   * Tạo vector $Z_{\text{melody}} \in \mathbb{R}^{32}$ ghép cùng $Z_{\text{audio}}$ và $Z_{\text{pause}}$.
2. **Xây dựng Bộ Đánh giá Cross-Validation:**
   * Script `tests/test_cross_validation.py` chạy 5-Fold Stratified CV và đánh giá Leave-One-Corpus-Out (LOCO).
3. **Hoàn thiện Cặp Dữ liệu Multimodal Manifest:**
   * Sinh manifest `data/manifests/multimodal_train.csv` gồm các trường: `audio_path`, `transcript` (sinh từ PhoWhisper hoặc có sẵn), `primary_label` (5 lớp), `sub_label` (11 lớp).
4. **Huấn luyện Mô hình Cross-Modal Gated Fusion:**
   * Tối ưu hóa `HierarchicalMERModel` trong `src/fusion/train_multimodal.py` với Dual Heads: Head 1 (5 lớp chính) từ $Z_{\text{fused}}$, Head 2 (11 lớp con) từ $Z_{\text{semantic}}$.
   * Thử nghiệm cơ chế cửa sổ trượt Chunking 6.0s (overlap 50%) và Sigmoid Multi-label cho các đoạn độc thoại dài.
5. **Dựng Web App Demo / Thu thập Nhỏ:**
   * Giao diện Streamlit / Web nhỏ ghi âm giọng nói, trực quan hóa biểu đồ Emotional Timeline & Radar Chart, phục vụ báo cáo và thu thập thêm mẫu.
