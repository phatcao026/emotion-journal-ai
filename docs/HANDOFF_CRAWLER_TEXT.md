# TÀI LIỆU BÀN GIAO TOÀN DIỆN: PIPELINE CRAWLER NHẬT KÝ & TÂM SỰ ĐA PHƯƠNG THỨC (CRAWLER & CURATION SUITE)
## BÀN GIAO CHO THÀNH VIÊN PHỤ TRÁCH PHÂN HỆ XỬ LÝ NGÔN NGỮ (TEXT SUBSYSTEM)

* **Dự án:** Hệ thống Nhật ký Tâm lý & Nhận diện Cảm xúc Đa phương thức (Emotion Journal AI)
* **Thành phần bàn giao:** Bộ công cụ Thu thập Dữ liệu (Crawler Pipeline), Bộ lọc Chất lượng Chuyên sâu (JournalFilter v2), Bộ quản lý Lưu trữ & Cách ly Khủng hoảng (CrawledDatasetManager), và Công cụ Gán nhãn Kiểm định (Annotation Tool).
* **Workspace:** `d:\TravisL\emotion-journal-ai`
* **Nhánh Git:** `feature/crawler-journal-pipeline`
* **Ngày cập nhật:** 2026-10-06
* **Trạng thái Kiểm thử:** **114/114 Unit Tests Passed (100%)**

---

## 1. TỔNG QUAN KIẾN TRÚC & MỤC TIÊU BÀN GIAO

Để giải quyết bài toán Nhận diện Cảm xúc trong Nhật ký (Journaling Emotion Recognition) mà không bị loãng bởi các dữ liệu bình luận mạng xã hội dạo (như ViGoEmotions), hệ thống crawler chuyên biệt này được xây dựng nhằm thu thập **các bài viết tự sự cá nhân gốc (First-person personal disclosures)** từ Meta Threads và VnExpress Tâm sự, đồng thời trích xuất âm thanh độc thoại giọng người từ YouTube/TikTok Radio.

```
                      [ WEB / THREADS / VNEXPRESS / YOUTUBE ]
                                         │
                                         ▼
                             [ PLAYWRIGHT CRAWLER ]
                                         │
                                         ▼
                               [ JOURNALFILTER V2 ]
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
         ▼                               ▼                               ▼
  [ LOẠI BỎ RÁC ]              [ CÁCH LY NGUY HIỂM ]            [ CHẤP NHẬN DỮ LIỆU ]
  - Chợ vé / Scalping          - Nguy cơ tự hại                 - Text tự sự chuẩn mực
  - PR Showbiz / Tin tức       - Ý định tự sát                  - Voice 16kHz độc thoại
  - Quảng cáo / Bán hàng       -> data/crawled/                 - Text-Voice đa phương thức
                                  crisis_quarantine/            -> data/crawled/{text, voice, text_voice}
                                                                         │
                                                                         ▼
                                                            [ ANNOTATION TOOL (STREAMLIT) ]
                                                            - Duyệt & hiệu chỉnh nhãn
                                                            - Tính Fleiss' Kappa / Cohen's Kappa
                                                                         │
                                                                         ▼
                                                            [ TEXT SUBSYSTEM (PHOBERT / NLP) ]
```

---

## 2. BỘ LỌC CHẤT LƯỢNG CHUYÊN SÂU: JOURNALFILTER V2 (QUALITY GATES)

Bộ lọc `JournalFilter` tại [scripts/crawler/journal_filter.py](file:///d:/TravisL/emotion-journal-ai/scripts/crawler/journal_filter.py) đã được đồng bộ chặt chẽ với tiền xử lý NLP của dự án và trải qua kiểm chứng thực nghiệm 100% trên dữ liệu cào thật:

### 2.1. Đồng bộ Tiền xử lý (TextPreprocessor Alignment)
* Tự động gọi [VietnameseTextPreprocessor](file:///d:/TravisL/emotion-journal-ai/src/text/text_preprocessor.py):
  - Chuẩn hóa Unicode NFC toàn diện tiếng Việt.
  - Xử lý bảng teencode chuẩn (`ko` -> `không`, `dc` -> `được`, `bt` -> `biết`, `hqua` -> `hôm qua`).
  - Khử nhiễu ký tự zero-width, thẻ ASR artifacts, chuẩn hóa khoảng trắng.
* Tự động cắt bỏ các tiền tố giao diện UI của Threads: header thời gian (`27/04/2026`, `3 giờ trước`, `5 ngày`), huy hiệu tác giả (`· Tác giả`), nút điều hướng (`Xem thêm`, `Dịch`, `Trả lời`).

### 2.2. Kiểm định Ngôi xưng Tự sự (First-Person Gatekeeper)
* Bắt buộc có đại từ tự xưng ngôi thứ nhất hoặc chỉ dấu nội tâm:
  - Đại từ chuẩn mực: `tôi`, `mình`, `tớ`, `em`, `anh`, `ta`, `bản thân`, `lòng mình`, `trong lòng`.
  - Đại từ thân mật MXH: `tui`, `toi`, `mk`, `mik`, `mềnh`, `tao`, `t`, `e`.
* **Cơ chế chống nhận diện nhầm tin PR Showbiz:** Phân biệt `anh/chị` khi đứng trước tên riêng viết hoa (ví dụ: *"chúc mừng anh HIEUTHUHAI"*) là danh xưng ngôi thứ ba trong bài tin tức truyền thông, không cho phép vượt qua cổng tự sự cá nhân.

### 2.3. Bộ lọc Hành vi Chợ vé / Giao dịch (Transactional Scalping vs. Emotional Experience)
* **Kết quả thực nghiệm đã kiểm chứng:** Các sự kiện/hoạt động đời sống như **đi concert, tập gym, thể thao, ngày đầu đi làm, xa nhà** mang lại cảm xúc cá nhân rất mạnh (tự ti ngoại hình, áp lực đồng trang lứa, mâu thuẫn tình cảm với người yêu, xúc động rơi nước mắt khi nghe bài hát gắn với thanh xuân).
* **Quy tắc phân biệt:** Không cấm theo chủ đề; thay vào đó, triệt tiêu 100% hành vi giao dịch vé chợ đen thông qua Regex chuyên sâu:
  - `(?:pass|tìm|cần\s+tìm|cần\s+pass|mua|nhượng|trade|đổi)\s+(?:\d+\s+)?(?:vé|slot|tix|ticket|hạng|ghế|merch)`
  - `(?:vé\s+)?cần\s+pass|ai\s+pass|pass\s+nhanh|chốt\s+nhanh|inbox\s+mình|ib\s+mình|nhượng\s+lại`
  - `(?:giá\s+vé|wts|wtb|gdtt|gdtg|full\s+merch|fast\s+track|check\s+legit|sold\s+out|sold\s+hết|presale|general\s+sale|hồi\s+máu|camp\s+vé|tiền\s+camp)`

### 2.4. Nhận diện Phạm vi Phủ định & Hiệu chỉnh Taxonomy
* **Negation Scope Detection:** Khi từ chỉ cảm xúc tích cực đi liền sau từ phủ định (`không vui`, `chẳng an tâm`, `chưa hạnh phúc`), hệ thống tự động đảo chiều để tăng điểm `SADNESS` / `ANXIETY`, ngăn chặn hiện tượng gán nhầm thành `JOY`.
* **Hiệu chỉnh Phân loại Cảm xúc (Taxonomy Calibration):** Rút từ `gia đình` đơn lẻ ra khỏi nhóm nhãn `CONNECTION` -> `JOY` (chỉ kích hoạt khi có từ khóa tình cảm như `gia đình ấm áp`, `yêu gia đình`), tránh việc các bài rao vé hoặc câu chuyện ly hôn bi kịch có nhắc chữ "gia đình" bị gán sai nhãn tích cực.

### 2.5. Bảo mật PII & Cách ly Khủng hoảng (Crisis Quarantine)
* Tự động ẩn danh hóa PII: Che số điện thoại (`[PHONE]`), email (`[EMAIL]`), link web (`[URL]`).
* Cách ly các mẫu có ý định tự hại (`tự tử`, `muốn chết`, `kết thúc cuộc đời`, `rạch tay`) vào riêng thư mục [data/crawled/crisis_quarantine/](file:///d:/TravisL/emotion-journal-ai/data/crawled/crisis_quarantine/) để bảo vệ mô hình Text không học phải các mẫu tiêu cực độc hại mà không có giám sát y tế.

---

## 3. CẤU TRÚC DỮ LIỆU & HƯỚNG DẪN TÍCH HỢP CHO NHÓM TEXT

### 3.1. Cấu trúc Lưu trữ trên Ổ đĩa
Dữ liệu cào được tổ chức trong [data/crawled/](file:///d:/TravisL/emotion-journal-ai/data/crawled/):
```
data/crawled/
├── metadata.csv                 # Bảng chỉ mục tổng hợp toàn bộ dataset
├── metadata.jsonl               # Stream metadata định dạng JSON Lines
├── text/                        # Chứa các file .json bài viết văn bản tâm sự thuần
├── voice/                       # Chứa file .json metadata và audio .wav 16kHz
├── text_voice/                  # Chứa các cặp song song audio 16kHz + text transcript
└── crisis_quarantine/           # Thư mục cách ly bài viết khủng hoảng tâm lý
```

### 3.2. Cấu trúc Một Mẫu Dữ liệu Text (.json)
Mỗi mẫu văn bản trong `data/crawled/text/<sample_id>.json` có schema chuẩn mực như sau:
```json
{
  "sample_id": "threads_search_txt_8f12a3b4",
  "text": "Đêm muộn ngồi một mình trong phòng trọ, dạo này áp lực công việc khiến mình thấy ngột ngạt và cô đơn vô cùng...",
  "primary_label": "SADNESS",
  "sub_label": "LONELINESS",
  "emotion_score": 0.88,
  "modality": "text",
  "author": "an_danh",
  "author_hash": "e3b0c44298fc",
  "source_platform": "threads_search",
  "source_url": "https://www.threads.net/search?q=...",
  "narrative_score": 0.8,
  "matched_pronouns": ["mình"],
  "matched_reflective": ["đêm muộn", "một mình", "áp lực", "cô đơn"],
  "matched_keywords": ["ngột ngạt", "cô đơn"],
  "crawled_at": "2026-10-06T20:55:00"
}
```

### 3.3. Đoạn mã Python Mẫu để Nhóm Text Đọc Dữ liệu vào Pipeline Huấn luyện

Nhóm Text có thể nạp dữ liệu trực tiếp bằng `CrawledDatasetManager` hoặc pandas:

```python
from pathlib import Path
import pandas as pd
from scripts.crawler.dataset_manager import CrawledDatasetManager

# Cách 1: Nạp qua Dataset Manager (tự động loại trừ crisis quarantine)
dm = CrawledDatasetManager(base_dir="data/crawled")
df = dm.get_dataframe()

# Lọc riêng tập dữ liệu Text
text_df = df[df["modality"].isin(["text", "text_voice"])].copy()
print(f"Tổng số mẫu văn bản khả dụng: {len(text_df)}")
print(text_df[["sample_id", "primary_label", "sub_label", "author_hash"]].head())

# Cách 2: Đọc trực tiếp từ file metadata CSV/JSONL
# df = pd.read_csv("data/crawled/metadata.csv")
```

### 3.4. YÊU CẦU BẮT BUỘC: Participant-Aware Split (Chia tập chống rò rỉ tác giả)
> [!IMPORTANT]
> Khi chia tập **Train / Validation / Test** cho mô hình phân loại Text (PhoBERT / Transformer), **BẮT BUỘC PHẢI CHIA THEO `author_hash`** (sử dụng `GroupKFold` hoặc `StratifiedGroupKFold` theo `author_hash`).
> **Lý do:** Một tác giả có thể viết nhiều bài nhật ký trên Threads hoặc VnExpress. Nếu bài của cùng một người vừa xuất hiện ở Train vừa ở Test, mô hình sẽ học thuộc phong cách hành văn của cá nhân đó (Overfitting Speaker/Author Style) thay vì học biểu hiện cảm xúc, dẫn tới hiện tượng rò rỉ dữ liệu (Data Leakage) nghiêm trọng.

---

## 4. MA TRẬN TỪ KHÓA TÌM KIẾM ĐÃ XÁC THỰC (VALIDATED KEYWORD MATRIX)

Khi cần mở rộng thêm quy mô dữ liệu cho nhóm Text, sử dụng ma trận từ khóa đã kiểm chứng trong `scripts/crawler/crawler_threads.py`:

| Nhóm Từ khóa | Từ khóa kiểm chứng | Đặc tính cảm xúc thu được |
| :--- | :--- | :--- |
| **Hoạt động đời sống & Cột mốc cá nhân** | `tập gym` | Tự ti ngoại hình, bối rối phương hướng, mâu thuẫn người yêu vì tập luyện. |
| | `concert` | Giải tỏa muộn phiền, vỡ òa hạnh phúc gặp thần tượng, hoài niệm thanh xuân. |
| | `hành trình` | Nhật ký nỗ lực giảm cân, học tập, kiên trì vượt qua khó khăn. |
| | `ngày đầu tiên đi làm`, `xa nhà`, `thay đổi bản thân` | Bỡ ngỡ, lạc lõng nơi đất khách, áp lực hòa nhập môi trường mới. |
| **Trạng thái tâm lý & Cảm xúc** | `áp lực`, `áp lực đồng trang lứa` | Trăn trở tài chính, so sánh thành công với bạn bè, hoang mang tương lai. |
| | `một mình`, `cô đơn` | Nỗi cô độc sinh viên, sự trống trải cuối tuần, tự lập vượt nghịch cảnh. |
| | `chữa lành` | Tìm lại bình yên, buông bỏ áp lực quá khứ. |
| **Văn phong nhật ký & Tự sự** | `tâm sự` | Tâm sự gia đình, chuyện công việc, ovtk, trầm cảm nhẹ cần người lắng nghe. |
| | `nhật ký` | Nhật ký biết ơn, phản ánh thói quen viết nội tâm cá nhân. |
| | `hôm nay của mình`, `chợt nhận ra` | Trải lòng đời thường sâu sắc. |

---

## 5. CÔNG CỤ GÁN NHÃN BÁN TỰ ĐỘNG & ĐO ĐỘ ĐỒNG THUẬN (ANNOTATION TOOL)

Để nhóm Text và nhóm chú thích dữ liệu kiểm định lại chất lượng nhãn trước khi huấn luyện:
* **Khởi chạy ứng dụng Web:**
  ```bash
  python run_annotation.py
  # hoặc:
  streamlit run scripts/annotation_tool/app.py
  ```
* **Tính năng:**
  1. Giao diện trực quan xem nội dung bài viết, nghe audio (đối với mẫu voice), xem các từ khóa nhận diện.
  2. Nút chấp nhận nhãn gợi ý hoặc chỉnh sửa nhãn (`Primary` và `Sub-emotion`).
  3. Đánh dấu loại bỏ (`Reject`) nếu phát hiện bài viết quảng cáo tinh vi lọt lưới.
  4. Báo cáo đo lường độ tin cậy liên người gán nhãn: **Fleiss' Kappa** (nhiều người) và **Cohen's Kappa** (cặp người) theo đúng tiêu chuẩn khoa học.

---

## 6. HƯỚNG DẪN CHẠY KIỂM THỬ (TEST VERIFICATION)

Toàn bộ hệ thống được bảo vệ bởi test suite tự động. Trước khi merge hoặc triển khai:

```powershell
# Chạy toàn bộ 114 bài kiểm thử
.\.venv\Scripts\pytest.exe tests/

# Chạy riêng bài kiểm thử pipeline crawler & journal filter
.\.venv\Scripts\pytest.exe tests/test_crawler_pipeline.py -v
```

**Kết quả kiểm thử hiện tại:**
```
tests/test_annotation_manager.py ...                                     [  2%]
tests/test_audio_pipeline.py .......................                     [ 22%]
tests/test_crawler_pipeline.py .................                         [ 37%]
tests/test_cross_validation.py .......                                   [ 43%]
tests/test_crosscorpus_pipeline.py ....                                  [ 47%]
tests/test_multimodal_pass.py .....................                      [ 65%]
tests/test_text_pipeline.py ................                             [ 79%]
tests/test_voice_pipeline.py .......................                     [100%]
============================ 114 passed in 19.44s =============================
```
