# Bộ Công Cụ Thu Thập & Chuẩn Hóa Dữ Liệu Đa Phương Thức (Emotion Journal AI Crawler Suite v2)

Hệ thống chuyên dụng thu thập, lọc và chuẩn hóa dữ liệu đa phương thức (**Text + Voice**, Video mở rộng) phục vụ bài toán **Nhận diện Cảm xúc Đa phương thức (Multimodal Emotion Recognition - MER)** cho ứng dụng **Nhật ký Tự sự Sức khỏe Tinh thần**.

---

## 1. Kiến Trúc & Các Công Cụ Trong Bộ Thu Thập

| Tên Script | Chức năng chính | Nguồn dữ liệu |
| :--- | :--- | :--- |
| **[`run_crawler.py`](file:///d:/TravisL/emotion-journal-ai/scripts/crawler/run_crawler.py)** | **Bộ điều khiển trung tâm (Unified CLI Runner)**: Cung cấp giao diện dòng lệnh duy nhất cho toàn bộ các chế độ (`news`, `voice`, `threads`, `status`, `balance`). | Quản lý toàn bộ pipeline |
| **[`crawler_news_journal.py`](file:///d:/TravisL/emotion-journal-ai/scripts/crawler/crawler_news_journal.py)** | Thu thập bài viết **tâm sự / nhật ký cuộc đời dài có chiều sâu** từ báo chính thống (VnExpress Tâm sự qua RSS & Web), văn phong chuẩn mực và tuyệt đối an toàn chính trị. | VnExpress Tâm sự |
| **[`crawler_voice_radio.py`](file:///d:/TravisL/emotion-journal-ai/scripts/crawler/crawler_voice_radio.py)** | Thu thập **VOICE độc thoại cảm xúc** từ các kênh Radio/Podcast tâm sự. Hỗ trợ **PhoWhisper ASR** sinh transcript thật cho `text_voice/`, tùy chọn **Demucs** bóc tách nhạc nền, chuẩn hóa 16kHz Mono. | YouTube, TikTok, Podcast, File local |
| **[`crawler_threads.py`](file:///d:/TravisL/emotion-journal-ai/scripts/crawler/crawler_threads.py)** | Quét tự động theo **Trang cá nhân** hoặc **Từ khóa tìm kiếm** trên Threads (`"tâm sự"`, `"nhật ký"`, `"áp lực"...`), bóc tách audio và văn bản. | Threads Search / Profiles |
| **[`crawler_thread_post.py`](file:///d:/TravisL/emotion-journal-ai/scripts/crawler/crawler_thread_post.py)** | Cào toàn bộ **bài viết & bình luận tâm sự / confession** từ một URL Threads cụ thể. Tự động bypass popup đăng nhập, trích xuất dữ liệu JSON nhúng và Relay GraphQL. | Threads Post permalink |
| **[`journal_filter.py`](file:///d:/TravisL/emotion-journal-ai/scripts/crawler/journal_filter.py)** | **Bộ lọc thông minh v2**: Kiểm định ngôi xưng tự sự (`tôi`, `mình`, `em`), lọc spam, xử lý **phạm vi phủ định (Negation Scope)**, **phát hiện nguy cơ khủng hoảng/tự hại (Crisis Detector)**, và ẩn danh PII. | Bộ lọc dùng chung |
| **[`voice_processor.py`](file:///d:/TravisL/emotion-journal-ai/scripts/crawler/voice_processor.py)** | Chuẩn hóa âm thanh 16kHz Mono PCM 16-bit, kiểm định RMS, peak clipping, VAD speech presence, phân đoạn monologue và hỗ trợ Demucs vocal isolation. | Tiền xử lý âm học |
| **[`dataset_manager.py`](file:///d:/TravisL/emotion-journal-ai/scripts/crawler/dataset_manager.py)** | Quản lý lưu trữ chuẩn hóa, băm mã tác giả `author_hash` (Participant-aware split), cách ly ca khủng hoảng, khử trùng lặp URL và xuất báo cáo thống kê. | Quản lý dữ liệu |

---

## 2. Hướng Dẫn Sử Dụng Nhanh (Qua Bộ Điều Khiển Trung Tâm `run_crawler.py`)

### A. Kiểm tra Dashboard Thống kê & Độ Lệch Nhãn
```powershell
# Xem tổng số mẫu, phân bố 5 Primary & 11 Sub emotions, số người nói độc lập:
python scripts/crawler/run_crawler.py status

# Phân tích độ lệch giữa nhóm Cảm xúc Tích cực và Tiêu cực, nhận gợi ý từ khóa bù trừ:
python scripts/crawler/run_crawler.py balance
```

---

### B. Cào Bài Viết Tự Sự Từ Báo Chí Chính Thống (VnExpress Tâm Sự)
Nguồn bài viết chất lượng cao nhất, an toàn chính trị tuyệt đối:
```powershell
# Cào 15 bài tâm sự mới nhất từ RSS và Web:
python scripts/crawler/run_crawler.py news --max-items 15 --min-words 40
```

---

### C. Thu Thập Dữ Liệu VOICE & Cặp Đa Phương Thức (Text-Voice)
```powershell
# 1. Xem danh sách kênh Radio/Podcast gợi ý:
python scripts/crawler/run_crawler.py voice --list-sources

# 2. Tải 1 tập radio, tự động cắt thành các clip 45s chuẩn 16kHz Mono:
python scripts/crawler/run_crawler.py voice `
  --url "https://www.youtube.com/watch?v=..." `
  --primary SADNESS `
  --sub LONELINESS `
  --author "nguoigiukinhiem"

# 3. Tải kèm PhoWhisper ASR để sinh transcript thật cho cặp đôi text_voice/:
python scripts/crawler/run_crawler.py voice `
  --url "https://www.youtube.com/watch?v=..." `
  --primary JOY `
  --sub CALM `
  --transcribe

# 4. Tùy chọn bóc tách nhạc nền radio (Vocal Isolation bằng Demucs):
python scripts/crawler/run_crawler.py voice `
  --file "path/to/radio_episode.wav" `
  --remove-music `
  --primary SADNESS
```

---

### D. Quét Tìm Kiếm Bài Viết Tự Sự Trên Threads
```powershell
# Tìm kiếm các chủ đề tâm sự & nhật ký:
python scripts/crawler/run_crawler.py threads `
  --search "tâm sự" "nhật ký hôm nay" "áp lực cuộc sống" "bình yên" `
  --max-items 15
```

---

## 3. Chiến Lược Phân Tầng Dữ Liệu Huấn Luyện (Training Tiers Architecture)

Nhằm tối ưu hóa độ chính xác và tính thực tiễn lâm sàng cho mô hình Stage 2 & Stage 3 Multimodal Emotion Recognition (MER), toàn bộ dữ liệu cào được gắn nhãn phân tầng minh bạch trong `metadata.csv` và `metadata.jsonl`:

| Phân tầng (Data Tier) | Bản chất Dữ liệu | Tiêu chuẩn Đánh giá | Vai trò Huấn luyện Hạ nguồn |
| :--- | :--- | :--- | :--- |
| **Tier 1: `deep_journal`** | **Nhật ký & Tâm sự trải lòng chuyên sâu** (First-Person Affective Self-Disclosure) | Ngôi thứ nhất (`tôi`, `mình`, `em`), bộc bạch cảm xúc chân thật, giàu từ ngữ nội tâm (`nghẹt thở`, `bật khóc`, `áp lực`, `cô đơn`, `ấm lòng`), không có văn phong thuyết giảng. | **Nòng cốt để huấn luyện chuyên sâu (Fine-tune Stage 2/3)**: Học các sắc thái cảm xúc lâm sàng, mô hình hóa sự bất đồng bộ giữa Lời nói và Giọng nói (Affective Masking). |
| **Tier 2: `general_monologue`** | **Độc thoại sạch / Podcast chia sẻ / Tản văn** (General Monologue MER) | Âm học đạt chuẩn ITU-T P.56 SNR >= 12dB, EBU R128 RMS, không nhạc nền, giọng phát thanh/giảng bài hướng dẫn (`chúng ta`, `bài học`, `lời khuyên`). | **Dữ liệu bổ trợ cho MER cơ bản**: Huấn luyện biểu diễn âm học nền tảng (Stage 1.5 SER, speech SSL tuning, melody/pause prior baseline), mở rộng số lượng speaker clusters (`author_hash`) mà không làm loãng tập nhật ký chuyên sâu. |
| **Quarantine: `crisis_quarantine`** | **Nguy cơ Khủng hoảng & Tự hại** (Bioethics & Clinical Safety) | Khớp từ khóa nguy cấp (theo thang đo C-SSRS: tự tử, muốn chết, rạch tay...). | **Cách ly an toàn tuyệt đối**: Không đưa vào tập train thông thường; bảo đảm quyền riêng tư và đạo đức y sinh. |

---

## 4. Cấu Trúc Dữ Liệu Đầu Ra Chuẩn Hóa

Dữ liệu được lưu trữ tự động tại thư mục `data/crawled/`:
```text
data/crawled/
├── text/                  # Các mẫu văn bản tâm sự / nhật ký hợp lệ (.json)
├── voice/                 # Các mẫu âm thanh độc thoại 16kHz Mono (.wav + .json)
├── text_voice/            # Cặp đôi đa phương thức hoàn chỉnh (.wav + .json transcript thật)
├── crisis_quarantine/     # KHU VỰC CÁCH LY: Các bài viết chứa nguy cơ tự hại/khủng hoảng tâm lý
├── metadata.csv           # Bảng chỉ mục metadata tổng hợp (kèm data_tier, author_hash, is_crisis, voice_type)
└── metadata.jsonl         # Bảng JSONL metadata dạng stream (hỗ trợ resumable cào nối tiếp)
```

---

## 5. Chạy Kiểm Thử Tự Động (Unit Tests)

```powershell
python -m unittest tests/test_crawler_pipeline.py -v
```
Bộ test tự động (24 test cases) kiểm định:
- Phân tầng chính xác `deep_journal` vs `general_monologue` qua `AffectiveSelfDisclosureFilter`.
- Nhận diện phạm vi phủ định (*"không vui"* $\rightarrow$ không bị gán nhầm `JOY`).
- Phát hiện và cách ly ca khủng hoảng vào `crisis_quarantine/`.
- Cơ chế băm danh tính tác giả `author_hash` (chống rò rỉ người nói).
- Cơ chế chống cào trùng lặp URL (Resumable Frontier).
- Khử thông tin cá nhân PII (số điện thoại, email).
- Kiểm định âm học chuẩn 16kHz Mono WAV, SNR ITU-T P.56, EBU R128 RMS, 0.000% clipping.
