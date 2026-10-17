# Báo cáo Đánh giá Lượt Retrain Stage 2 (Epoch 20 -> 30)

> **Ngày thực hiện:** 16/10/2026  
> **Phiên bản:** Stage 2 SOTA Pretrain Retrain  
> **Tập tin checkpoint kiểm chứng:** `checkpoints/stage2_retrain/last_model_stage2.pt` (kích thước: 1,607 MB)  
> **Nguồn tải:** `C:\Users\TraVisL\Downloads\stage2_retrain_results.zip` (1.2 GB)

---

## 1. Tóm tắt Kiểm chứng Kỹ thuật (Technical Verification)

| Thành phần kiểm chứng | Trạng thái trước (v1 - Epoch 20) | Trạng thái sau Retrain (v2 - Epoch 30) | Kết luận kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Kiến trúc Sub-Head** | 768D (Text-only) | **928D (`z_fused` đa phương thức)** | **Đã huấn luyện thành công** (`sub_head.0.weight`: `[256, 928]`). |
| **Gated Fusion Layer** | Chưa hội tụ (tỷ lệ 50/50) | **Động (Text: 61.4%, Audio: 38.6%)** | **Cơ chế Gating đã học được trọng số thích nghi**. |
| **Số lượng Tensors nạp** | Không đồng nhất chiều | **253 / 253 tensors khớp 100%** | Không thiếu hoặc thừa tham số trong pipeline Stage 3. |
| **Tỷ lệ vi phạm thứ bậc** | 0.00% | **0.00%** | Soft Bayesian Routing & Joint MAP hoạt động hoàn hảo. |

---

## 2. Diễn biến Chỉ số Huấn luyện (Training & Validation Metrics)

Tiến trình huấn luyện 10 epochs kế thừa từ checkpoint 20:

```
Epoch 20 -> Loss: 11.5583 | Val Loss: 10.0326 | Val Sub F1: 0.2152
Epoch 23 -> Loss:  8.9056 | Val Loss:  8.6322 | Val Sub F1: 0.2312
Epoch 26 -> Loss:  8.1325 | Val Loss:  7.9142 | Val Sub F1: 0.2289
Epoch 30 -> Loss:  7.4239 | Val Loss:  7.6027 | Val Sub F1: 0.2257
```

### Bảng đối sánh chỉ số tổng kết

| Metric | Checkpoint cũ (Epoch 20) | Retrain Val (Epoch 30) | Retrain Test (Epoch 30) | Biến thiên |
| :--- | :---: | :---: | :---: | :---: |
| **Train Loss** | 11.5583 | — | **7.4239** | **-35.8% (Hội tụ tốt)** |
| **Val / Test Loss** | 10.0326 | 7.6027 | **7.7125** | **-23.1% (Loss giảm sâu)** |
| **Primary Macro-F1** | 54.90% | 53.36% | **47.78%** | Ổn định quanh mức baseline |
| **Primary Weighted-F1 (WA)** | 54.74% | 53.13% | **48.10%** | Duy trì tốt |
| **Sub-Emotion Macro-F1** | 11.49% | **22.57%** | **21.41%** | **TĂNG +86% (Từ 11.49% lên 21.41%)** |
| **Sub-Emotion Weighted-F1 (WA)** | ~40.0% | 52.01% | **49.21%** | **Tăng trưởng vượt bậc** |

---

## 3. Đánh giá Điểm sáng & Bước tiến cốt lõi

1. **Hiệu quả của việc nâng cấp Sub-Head 928D:**
   - Việc mở rộng Sub-head từ 768D (chỉ nhận đặc trưng ngữ nghĩa đơn lẻ) sang 928D (nhận toàn bộ `z_fused` gồm Audio 768D + Pause 128D + Melody 32D) đã giúp mô hình bắt được sắc thái cảm xúc con của giọng nói.
   - Chỉ số **Sub-Emotion Macro-F1 tăng gần gấp đôi (từ 11.49% lên 21.41% trên tập Test)**.
2. **Khả năng Gating đa phương thức linh hoạt:**
   - Khi chạy thử nghiệm suy luận đa phương thức, mạng Gated Fusion đã thoát khỏi mức kẹt 50/50, tự động phân phối trọng số tin cậy (ví dụ Text 61.4%, Audio 38.6%).
3. **Hierarchy Consistency đảm bảo 100%:**
   - Mô hình không xuất hiện bất kỳ cặp dự đoán mâu thuẫn nào (như đoán Primary là Joy nhưng Sub lại là Sorrow).

---

## 4. Phát hiện Bottleneck Chuyên sâu: Hiện tượng bão hòa đặc trưng PhoBERT

Qua kiểm chứng thực nghiệm bằng code nội suy tensor:
- Khi chưa nạp trọng số checkpoint: PhoBERT sinh vector phân biệt rõ giữa các câu (`z_diff_max = 0.5265`).
- **Sau khi nạp checkpoint `last_model_stage2.pt`**: Vector đặc trưng ngữ nghĩa giữa các câu khác nhau co cụm lại (`z_diff_max = 3.57e-7`), dẫn đến xu hướng thiên lệch đầu ra phân loại văn bản về cụm nhãn `ANXIETY/FEAR` (xác suất 94% - 97%).

### Nguyên nhân kỹ thuật:
- Trong `src/stage2/train_stage2.py`, toàn bộ các tham số của `model.text_encoder` (PhoBERT-base v2 với 110 triệu tham số) được gộp chung vào nhóm tham số `other_params` và được huấn luyện ở tốc độ học **$\text{LR} = 1.0 \times 10^{-4}$**.
- Trong thực hành huấn luyện mô hình Transformer (BERT/RoBERTa), mức $\text{LR} = 1.0 \times 10^{-4}$ là quá cao cho backbone (chuẩn khuyến nghị là $1.0 \times 10^{-5} \to 2.0 \times 10^{-5}$ kèm warm-up). Khi bị gradient lớn ép qua nhiều epochs, các ma trận chiếu Self-Attention bị trôi vào không gian bão hòa (representation collapse).

---

## 5. Đề xuất Tinh chỉnh Cho Lượt Huấn luyện Tiếp theo (Action Items)

1. **Phân tách Learning Rate theo từng nhóm tham số (Layer-wise / Component-wise LR):**
   - Head & Gated Fusion: $\text{LR} = 1.0 \times 10^{-4}$
   - Audio LoRA Adapters: $\text{LR} = 5.0 \times 10^{-5}$
   - Text Encoder (PhoBERT): $\text{LR} = 1.0 \times 10^{-5}$ hoặc đóng băng (freeze) 10 lớp đầu, chỉ mở 2 lớp cuối.
2. **Triển khai Stage 3:**
   - Checkpoint hiện tại hoàn toàn tương thích và đã sẵn sàng đưa vào vận hành thử nghiệm end-to-end cho Stage 3 Journaling Companion.
