# Hướng Dẫn Sử Dụng Pipeline Dịch Dataset LE-motif Tiếng Việt

Bộ công cụ tự động hóa chuẩn bị dữ liệu tiếng Việt cho dataset **LE-motif** phục vụ bài toán Emotion Classification.

---

## 1. Cài đặt & Chuẩn bị API Key

1. Đăng ký và lấy **Gemini API Key** miễn phí tại [Google AI Studio](https://aistudio.google.com/apikey).
2. Thiết lập key bằng 1 trong các cách sau:
   - **Cách 1 (Khuyên dùng)**: Tạo file `.env` ở thư mục gốc của repo:
     ```env
     GEMINI_API_KEY=AIzaSy...
     ```
   - **Cách 2**: Đặt biến môi trường Windows (PowerShell):
     ```powershell
     $env:GEMINI_API_KEY="AIzaSy..."
     ```
   - **Cách 3**: Truyền trực tiếp qua tham số `--api-key`:
     ```powershell
     python scripts/prepare_lemotif_vi.py --api-key AIzaSy...
     ```

---

## 2. Các Bước Thực Hiện

### Bước 1: Chạy Thử Nghiệm Mẫu Pilot (30–50 câu) & Gửi Duyệt
Trích xuất 35 câu tiêu biểu (chứa từ phủ định, emoji, đa nhãn, cảm xúc hiếm như *Awkward, Jealous, Nostalgic*) để gửi lead duyệt tone dịch:

```powershell
python scripts/prepare_lemotif_vi.py --mode pilot --limit 35
```
*Kết quả xuất ra tại*: `data/raw/lemotif_vi_pilot.jsonl`

Kiểm tra tính hợp lệ của file pilot:
```powershell
python scripts/validate_lemotif_vi.py --file data/raw/lemotif_vi_pilot.jsonl
```

---

### Bước 2: Dịch Toàn Bộ 1,473 Câu
Sau khi phong cách dịch của pilot được duyệt, tiến hành dịch toàn bộ dataset:

```powershell
python scripts/prepare_lemotif_vi.py --mode full --batch-size 15 --delay 2.0
```
*Đặc điểm*:
- Có cơ chế **checkpointing** tự động lưu vào `data/raw/.checkpoint_lemotif_vi.jsonl`. Nếu mạng bị ngắt, bạn chỉ cần chạy lại lệnh trên, chương trình sẽ tự động dịch tiếp từ câu dang dở.
*Kết quả xuất ra tại*: `data/raw/lemotif_vi.jsonl`

---

### Bước 3: Kiểm Tra Kỹ Thuật & Lọc Mẫu Rà Soát Thủ Công (>= 10%)
Chạy script kiểm tra chất lượng:
```powershell
python scripts/validate_lemotif_vi.py --file data/raw/lemotif_vi.jsonl
```
*Script sẽ*:
- Kiểm tra 100% không trùng ID, không rỗng `text_vi`, giữ nguyên mảng nhãn gốc.
- Tự động xuất file `data/raw/lemotif_vi_review_samples.jsonl` (~150 mẫu, chiếm >= 10%) chứa các câu rủi ro cao (phủ định, emoji, đa nhãn, cảm xúc hiếm) để bạn mở ra rà soát nhanh bằng mắt.

---

### Bước 4: Tạo File Manifest Bàn Giao
Tạo file metadata `lemotif_vi_manifest.json` theo đúng quy chuẩn:
```powershell
python scripts/generate_manifest.py --translator "Tên_Của_Bạn"
```
*Kết quả xuất ra tại*: `data/raw/lemotif_vi_manifest.json`
