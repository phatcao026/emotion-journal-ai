# TÀI LIỆU ĐẶC TẢ YÊU CẦU PHẦN MỀM (SRS)
## Ứng dụng Nhật ký Cá nhân Đa phương thức có Pipeline AI Nghiên cứu
### Multimodal Personal Journal with an Experimental AI Analysis Pipeline

| Thuộc tính | Giá trị |
|---|---|
| **Phiên bản tài liệu** | 1.3 |
| **Ngày cập nhật** | 04/08/2026 |
| **Loại tài liệu** | Software Requirements Specification — bản nháp đồ án/nghiên cứu |
| **Trạng thái** | Draft — chưa baseline; output AI và ground truth chờ kết quả pilot |

## Lịch sử chỉnh sửa

| Phiên bản | Ngày | Nội dung |
|---|---|---|
| 1.0 | 29/07/2026 | Khởi tạo SRS từ yêu cầu sản phẩm ban đầu |
| 1.1 | 30/07/2026 | Bổ sung privacy, kiến trúc AI và tech stack ứng viên |
| 1.2 | 31/07/2026 | Bổ sung chat AI, khóa chỉnh sửa, cảnh báo và output nhãn cảm xúc |
| 1.3 | 04/08/2026 | Thu hẹp về journal-first và phân tích một entry; phân loại yêu cầu Committed/Research/Conditional/Future; tổng quát hóa output; bỏ chat AI, advice, trend, crisis detection và personalization khỏi phạm vi hiện tại; chuyển quyết định công nghệ sang SDD |

---

## 1. Giới thiệu

### 1.1 Mục đích

Tài liệu đặc tả yêu cầu cho một ứng dụng nhật ký cá nhân đa phương thức và một pipeline AI thử nghiệm ở cấp độ từng entry. Tài liệu làm cơ sở để:

- xác định phạm vi sản phẩm;
- thiết kế và kiểm thử các chức năng nhật ký cốt lõi;
- triển khai pilot nghiên cứu AI;
- tách các tính năng đã chốt khỏi tính năng phụ thuộc kết quả nghiên cứu;
- xây dựng SDD, API specification và research protocol ở giai đoạn tiếp theo.

### 1.2 Phạm vi hệ thống

Hệ thống cho phép người dùng tạo và quản lý nhật ký cá nhân bằng văn bản, giọng nói, hình ảnh và, nếu khả thi, video. Trải nghiệm cốt lõi là **journal-first**: người dùng viết, lưu, xem, sửa, xóa và xuất nhật ký như một ứng dụng nhật ký thông thường.

Hệ thống có một pipeline AI thử nghiệm để xử lý **dữ liệu của một entry hiện tại**. Pipeline có thể tạo candidate output liên quan đến sentiment, expressed emotion, valence/arousal, momentary affect hoặc reflection. Output chính, ground truth và cách biểu hiện trong giao diện chưa được chốt; chúng sẽ được quyết định sau pilot và validation.

Hệ thống trong phạm vi hiện tại:

- không phải chatbot trị liệu;
- không chẩn đoán bệnh hoặc tính symptom score;
- không phát hiện crisis/self-harm bằng emotion model;
- không tạo lời khuyên lâm sàng;
- không phân tích pattern/xu hướng qua nhiều entry;
- không dùng lịch sử để cá nhân hóa model;
- không cam kết multimodal luôn tốt hơn text-only.

### 1.3 Ngoài phạm vi tài liệu

Các nội dung sau thuộc tài liệu khác hoặc giai đoạn sau:

- UI/UX mockup chi tiết;
- lựa chọn Flutter/React Native hoặc framework cụ thể;
- API, ERD và deployment diagram chi tiết;
- model backbone, fusion architecture và hyperparameter cụ thể;
- research protocol hoàn chỉnh, power analysis và ethics application;
- chatbot đồng hành, lời khuyên, trend/pattern, personalization và crisis detection.

### 1.4 Đối tượng sử dụng tài liệu

- giảng viên hướng dẫn và hội đồng;
- nhóm phát triển;
- nhóm nghiên cứu/đánh giá AI;
- người phụ trách privacy và ethics review.

### 1.5 Quy ước trạng thái yêu cầu

| Trạng thái | Ý nghĩa |
|---|---|
| **Committed** | Yêu cầu sản phẩm cốt lõi sẽ được triển khai |
| **Research** | Chỉ triển khai trong pilot/nghiên cứu, có feature flag và consent phù hợp |
| **Conditional** | Chỉ phát hành cho người dùng nếu vượt tiêu chí nghiên cứu/UX |
| **Future** | Ngoài phạm vi phiên bản hiện tại |

---

## 2. Mô tả tổng quan

### 2.1 Bối cảnh sản phẩm

Các app hiện có cho thấy nhiều cách tiếp cận: mood check-in, nhật ký truyền thống, guided reflection và AI journal. Một số app đã hỗ trợ nhiều loại dữ liệu. Vì vậy, điểm nghiên cứu của hệ thống không phải là giả định “multimodal chưa từng tồn tại”, mà là kiểm tra:

> Trong bối cảnh nhật ký cá nhân tiếng Việt, mỗi modality đóng góp gì và loại output entry-level nào có thể được đánh giá đáng tin cậy?

Multimodal là giả thuyết nghiên cứu. Hệ thống phải so sánh với text-only và cho phép kết luận rằng một modality không tạo thêm giá trị.

### 2.2 Luồng sản phẩm cốt lõi

```text
Người dùng tạo một entry
  ├─ Text
  ├─ Voice
  ├─ Image
  └─ Video (Conditional)
        ↓
Xác nhận nội dung/transcript và lưu entry
        ↓
Mã hóa và lưu trữ
        ↓
Nếu người dùng thuộc pilot và đã consent:
  tạo AI processing job cho chính entry đó
        ↓
Lưu output kỹ thuật có version
        ↓
Chỉ hiển thị output nếu feature đã vượt research/UX gate
```

### 2.3 Actors và hệ thống liên quan

| Actor/hệ thống | Vai trò |
|---|---|
| **End User** | Tạo, quản lý, xem và xuất nhật ký |
| **Research Participant** | End User tự nguyện tham gia pilot và cung cấp loại reference đã consent |
| **Researcher/Admin** | Quản lý nghiên cứu, dataset version và model evaluation theo phân quyền |
| **Notification Service** | Gửi nhắc nhở và thông báo hoàn tất tác vụ |
| **AI Subsystem** | Thành phần nội bộ, không phải actor bên ngoài; xử lý entry khi được phép |

### 2.4 Môi trường vận hành

- Mobile: Android/iOS; nền tảng cụ thể được chốt trong SDD.
- Core journaling: hỗ trợ offline.
- AI processing: có thể on-device, server-side hoặc hybrid; quyết định thuộc SDD/threat model.
- Media processing: bất đồng bộ, không chặn thao tác viết và lưu nhật ký.

### 2.5 Giả định và phụ thuộc

- Người dùng chỉ cấp quyền camera/micro khi sử dụng modality tương ứng.
- Video có thể bị loại khỏi pilot đầu nếu không đủ dữ liệu, consent hoặc năng lực xử lý.
- Mọi dữ liệu nghiên cứu phải có consent theo mục đích và phiên bản.
- Dataset công khai chỉ dùng theo license và không tự được xem là ground truth cho nhật ký riêng tư.

---

## 3. Yêu cầu chức năng

### 3.1 Yêu cầu sản phẩm cốt lõi — Committed

| ID | Yêu cầu | Ưu tiên | Tiêu chí nghiệm thu tóm tắt |
|---|---|---|---|
| FR-C01 | Đăng ký/đăng nhập và quản lý phiên | Cao | Người dùng có thể tạo phiên, đăng nhập, đăng xuất và thu hồi phiên |
| FR-C02 | Tạo entry bằng text | Cao | Tạo, lưu và mở lại được entry text |
| FR-C03 | Ghi âm và chuyển voice thành transcript | Cao | Hiển thị transcript để người dùng xác nhận/sửa trước khi lưu |
| FR-C04 | Đính kèm hình ảnh | Trung bình | Cho phép thêm/xóa ảnh trước khi lưu; hiển thị trạng thái upload |
| FR-C05 | Đính kèm video | Thấp/Conditional | Chỉ bật khi SDD và pilot xác nhận khả thi; không cản trở text/voice/image |
| FR-C06 | Quản lý entry | Cao | Người dùng được xem, sửa và xóa entry; không có khóa chỉnh sửa 24 giờ |
| FR-C07 | Lưu lịch sử phiên bản entry | Trung bình | Khi entry đã từng được AI phân tích, hệ thống lưu `entry_version` để truy vết |
| FR-C08 | Ghi nhật ký offline và đồng bộ sau | Cao | Tạo/xem/sửa/xóa local khi offline; có trạng thái đồng bộ và xử lý xung đột |
| FR-C09 | Danh sách/timeline riêng tư | Cao | Mặc định không hiển thị nội dung nhạy cảm trên preview; người dùng điều chỉnh được |
| FR-C10 | Quản lý quyền theo modality | Cao | Camera/micro/media chỉ được dùng khi có quyền; người dùng có thể tắt riêng |
| FR-C11 | Xóa dữ liệu | Cao | Xóa entry/tài khoản kích hoạt deletion cascade theo retention policy |
| FR-C12 | Export/backup | Trung bình | Người dùng xuất được nội dung và media của mình ở định dạng được công bố |
| FR-C13 | Reminder | Thấp | Người dùng bật/tắt và chọn lịch; notification không chứa nội dung nhạy cảm |
| FR-C14 | Trạng thái tác vụ AI | Trung bình | Hiển thị queued/processing/completed/failed/cancelled khi AI được bật |
| FR-C15 | Nhật ký truy cập dữ liệu | Trung bình | Người dùng xem được các lần xử lý/chia sẻ quan trọng của dữ liệu của mình |

### 3.2 Yêu cầu nghiên cứu AI — Research

| ID | Yêu cầu | Ưu tiên | Tiêu chí nghiệm thu tóm tắt |
|---|---|---|---|
| FR-R01 | Modality adapters | Cao | Chuyển từng input thành representation và quality metadata phù hợp |
| FR-R02 | Phân tích một entry | Cao | Model không sử dụng entry lịch sử trong phạm vi nghiên cứu hiện tại |
| FR-R03 | Missing-modality support | Cao | Entry chỉ có một modality vẫn xử lý được; không bắt buộc đủ bốn modality |
| FR-R04 | Candidate output có cấu trúc | Cao | Lưu output type, schema version, score/label, confidence, evidence nếu có và trạng thái abstain |
| FR-R05 | Baseline và ablation | Cao | So sánh baseline, text/transcript-only và các tổ hợp modality trên cùng protocol |
| FR-R06 | Participant-aware evaluation | Cao | Không để entry của cùng participant ở cả train và locked test |
| FR-R07 | Pilot reference collection | Cao | Chỉ thu human annotation, user confirmation hoặc self-report theo protocol/consent đã chốt |
| FR-R08 | Research feedback | Trung bình | Người tham gia có thể xác nhận/sửa/từ chối output khi protocol yêu cầu |
| FR-R09 | Dataset/model versioning | Cao | Mọi run truy được dataset, split, preprocessing, model và metric version |
| FR-R10 | Model abstention | Cao | Model có thể không tạo kết luận khi input/uncertainty không đạt tiêu chí |

### 3.3 Yêu cầu phụ thuộc kết quả nghiên cứu — Conditional

| ID | Yêu cầu | Điều kiện mở khóa |
|---|---|---|
| FR-K01 | Hiển thị output AI cho người dùng | Output vượt tiêu chí validation và ethics/UX review |
| FR-K02 | Chuyển output kỹ thuật thành presentation | Presentation rules được version và kiểm thử riêng |
| FR-K03 | User confirmation tự nguyện | Nghiên cứu UX xác nhận không gây dẫn dắt/phiền quá mức |
| FR-K04 | Reflection định tính | Đạt tiêu chí factuality, unsupported-claim rate và usefulness đã chốt |

### 3.4 Chức năng ngoài phạm vi — Future

Các chức năng sau không phải yêu cầu của phiên bản hiện tại:

- chat AI theo phong cách “người bạn” và đặt tên AI;
- lời khuyên/động viên cá nhân hóa;
- pattern, memory và báo cáo xu hướng qua nhiều entry;
- personalization dựa trên lịch sử;
- symptom score, diagnosis hoặc treatment recommendation;
- AI-triggered crisis/self-harm detection.

Ứng dụng phải nói rõ rằng không thực hiện monitoring khủng hoảng thời gian thực. Nếu cung cấp tài nguyên hỗ trợ, tài nguyên đó phải do người dùng chủ động truy cập và không được trình bày như kết quả của một risk model chưa validation.

---

## 4. Use Case chính

### UC-01 — Tạo và lưu entry

- **Actor:** End User.
- **Tiền điều kiện:** Người dùng có phiên hợp lệ hoặc chế độ local được cho phép.
- **Luồng chính:**
  1. Người dùng tạo entry mới.
  2. Người dùng nhập text và tùy chọn voice/image/video đang được hỗ trợ.
  3. Với voice, hệ thống hiển thị transcript để xác nhận/sửa.
  4. Người dùng lưu entry.
  5. Hệ thống tạo `entry_version`, mã hóa và lưu local.
  6. Khi online, hệ thống đồng bộ theo chính sách đã cấu hình.
- **Luồng lỗi:** Thiếu quyền, file không hợp lệ, hết dung lượng hoặc sync thất bại phải có thông báo và cách thử lại.
- **Hậu điều kiện:** Entry lưu được kể cả khi AI không khả dụng.

### UC-02 — Phân tích entry trong research mode

- **Actor:** Research Participant, Researcher.
- **Tiền điều kiện:** Consent hợp lệ; entry có ít nhất một modality; feature flag nghiên cứu bật.
- **Luồng chính:**
  1. Hệ thống tạo processing job gắn với `entry_id` và `entry_version`.
  2. Mỗi modality được xử lý độc lập và ghi quality metadata.
  3. Model tạo candidate output theo `output_schema_version`.
  4. Hệ thống lưu model version, raw scores, confidence/calibration metadata, evidence nếu có và trạng thái abstain.
  5. Nếu protocol yêu cầu, participant cung cấp reference/feedback sau khi lưu entry.
- **Luồng lỗi:** Nếu một modality lỗi, hệ thống có thể tiếp tục với modality còn lại và ghi rõ coverage; không suy kết quả nếu không đạt quality threshold.
- **Hậu điều kiện:** Có output kỹ thuật có thể audit hoặc trạng thái abstained/failed.

### UC-03 — Sửa hoặc xóa entry

- **Actor:** End User.
- **Luồng chính:**
  1. Người dùng sửa hoặc xóa entry bất kỳ lúc nào theo policy.
  2. Khi sửa, hệ thống tạo phiên bản mới; kết quả AI cũ vẫn gắn với phiên bản cũ để audit nhưng không hiển thị như kết quả hiện hành.
  3. Khi xóa, hệ thống thực hiện deletion cascade cho nội dung, processing jobs và derived data theo consent/retention policy.
  4. Nếu snapshot đã vào dataset nghiên cứu, xử lý withdrawal theo research protocol.

### UC-04 — Quản lý consent nghiên cứu

- **Actor:** Research Participant.
- **Luồng chính:**
  1. Hệ thống giải thích mục đích, loại dữ liệu, nơi xử lý, thời hạn lưu và quyền rút lại.
  2. Người dùng opt-in riêng cho nghiên cứu; việc dùng app không phụ thuộc vào opt-in.
  3. Người dùng xem và rút consent.
  4. Hệ thống ghi consent version và áp dụng withdrawal policy.

### UC-05 — Hiển thị output đã được phê duyệt

- **Actor:** End User.
- **Tiền điều kiện:** FR-K01 được mở khóa; output không abstain; presentation version đã qua UX review.
- **Luồng chính:**
  1. Presentation layer nhận output kỹ thuật.
  2. Áp dụng presentation rules theo output type và uncertainty.
  3. Hiển thị câu reflection, thẻ, score hoặc lựa chọn “không hiển thị”, tùy phiên bản đã kiểm thử.
  4. Người dùng có thể xác nhận, từ chối hoặc ẩn kết quả nếu chức năng được bật.

---

## 5. Đặc tả output AI

### 5.1 Output kỹ thuật

SRS không cố định output thành emotion label. Một `AIAnalysisResult` tối thiểu gồm:

```text
result_id
entry_id
entry_version
output_type
output_schema_version
model_version_id
modality_coverage
quality_metadata
raw_scores_or_labels
confidence_and_calibration_metadata
evidence_references (optional)
abstained
abstention_reason
processing_status
created_at
```

Các `output_type` ứng viên:

- `sentiment`
- `expressed_emotion`
- `valence_arousal`
- `momentary_affect`
- `reflection`

Tên output phải khớp ground truth. Nếu không có self-report gần thời điểm, hệ thống không được gọi output là experienced/momentary affect.

### 5.2 Lớp trình bày

Output kỹ thuật và giao diện được version độc lập:

```text
AIAnalysisResult
        ↓
PresentationRuleVersion
        ↓
UI component
```

Presentation có thể là câu reflection, thẻ cảm xúc, thanh điểm, biểu tượng, câu hỏi xác nhận hoặc không hiển thị khi uncertainty cao. Không bắt buộc ba mức cao/trung bình/thấp trước khi calibration và UX study hoàn tất.

---

## 6. Yêu cầu phi chức năng

| ID | Nhóm | Yêu cầu có thể kiểm thử |
|---|---|---|
| NFR-01 | Privacy | Chức năng journal cốt lõi hoạt động mà không cần opt-in nghiên cứu |
| NFR-02 | Consent | Consent phải purpose-specific, versioned, revocable và tách khỏi điều khoản dùng app |
| NFR-03 | Encryption | Dữ liệu nhạy cảm được mã hóa khi lưu và khi truyền; thuật toán/key management được chốt trong SDD/threat model |
| NFR-04 | Logging | Application log, analytics và crash report không chứa raw journal/media/transcript |
| NFR-05 | Deletion | Xóa entry/tài khoản kích hoạt deletion cascade; thời hạn active store/backup phải được công bố và kiểm thử trước release |
| NFR-06 | Offline | Create/read/update/delete entry cốt lõi hoạt động offline; sync conflict không làm mất âm thầm dữ liệu |
| NFR-07 | Responsiveness | Lưu local một entry text phải hoàn thành P95 ≤ 2 giây trên test device profile được công bố |
| NFR-08 | AI latency | Text analysis target P95 ≤ 10 giây trong test profile; media job là async, có progress/status và không block journal flow |
| NFR-09 | Reliability | Processing job hỗ trợ idempotency, retry có giới hạn, cancellation và failure state rõ ràng |
| NFR-10 | Accessibility | Màn hình cốt lõi hỗ trợ dynamic text, screen reader và contrast theo chuẩn accessibility được chọn trong SDD |
| NFR-11 | Maintainability | Model/output/presentation có version và rollback; kết quả cũ truy được phiên bản đã tạo nó |
| NFR-12 | Research validity | Preprocessing, tuning, calibration và threshold không được nhìn locked test |
| NFR-13 | Participant split | Nếu có nhiều entry/người, locked test không chứa participant đã xuất hiện trong train |
| NFR-14 | Multimodal evaluation | Báo cáo text/transcript baseline và ablation trước khi claim fusion tạo thêm giá trị |
| NFR-15 | Uncertainty | Model được phép abstain; confidence chỉ hiển thị nếu có calibration và presentation đã kiểm thử |
| NFR-16 | Fairness | Báo hiệu năng/coverage theo subgroup khả thi như vùng miền, chất lượng ASR và loại modality; không claim đại diện quốc gia nếu sampling không hỗ trợ |
| NFR-17 | Security | Có threat model, phân quyền tối thiểu, secret rotation và quy trình xử lý sự cố trước production |

Các ngưỡng latency trên là target cho prototype và phải được xác nhận lại bằng test profile trong SDD. Chúng không được dùng để tuyên bố SLA production.

---

## 7. Mô hình dữ liệu tổng quan

| Thực thể | Mục đích | Thuộc tính chính rút gọn |
|---|---|---|
| `User` | Tài khoản và cài đặt | id, account metadata, privacy settings |
| `JournalEntry` | Entry logic | id, user_id, current_version, timestamps, sync status |
| `EntryVersion` | Phiên bản nội dung | id, entry_id, version, encrypted payload refs, created_at |
| `ContentAsset` | Text/voice/image/video | id, entry_version_id, modality, encrypted URI/payload, quality metadata |
| `AIProcessingJob` | Tác vụ AI | id, entry_version_id, status, retry count, timestamps |
| `AIAnalysisResult` | Output kỹ thuật | schema ở mục 5.1 |
| `PresentationRuleVersion` | Chuyển output sang UI | id, compatible output type/schema, version, status |
| `ConsentRecord` | Consent có version | participant_id, purpose, modalities, version, granted/withdrawn time |
| `ResearchReference` | Ground truth/reference | sample_id, reference_type, label/score, provenance, annotator protocol |
| `DatasetVersion` | Snapshot nghiên cứu | id, inclusion rules, consent scope, split version, checksum |
| `ModelVersion` | Phiên bản model | id, code/model checksum, training dataset, preprocessing version, license |
| `ModelEvaluation` | Kết quả đánh giá | model version, dataset/split, metrics, calibration, subgroup results |
| `DataAccessEvent` | Audit truy cập | actor, purpose, data category, timestamp, result |

Không có `Recommendation` hoặc `MoodTrend` trong mô hình hiện tại vì các tính năng đó ngoài phạm vi.

---

## 8. Privacy và kiến trúc xử lý AI

### 8.1 Nguyên tắc

SRS chỉ chốt outcome bảo mật, không chốt công nghệ. SDD phải chọn và mô tả một trong các hướng:

- on-device processing;
- controlled server-side processing;
- hybrid processing.

Nếu server có thể giải mã tạm để inference, hệ thống không được tuyên bố zero-knowledge hoặc “chỉ người dùng có thể đọc” theo nghĩa tuyệt đối. Threat model phải mô tả key flow, memory exposure, logs, backups, subprocessors và incident response.

### 8.2 Dữ liệu dẫn xuất

Transcript, embedding, feature vector, AI output và model feedback đều có thể chứa thông tin nhạy cảm. Chúng phải nằm trong retention/deletion/consent policy, không chỉ raw media.

### 8.3 Người thứ ba trong media

Image, voice và video có thể chứa người không phải chủ tài khoản. SDD/privacy review phải xác định cảnh báo, consent expectation, training exclusion và deletion behavior trước khi bật media contribution cho nghiên cứu.

### 8.4 Tuân thủ

Nhóm phải rà soát yêu cầu pháp lý hiện hành về dữ liệu cá nhân và dữ liệu nhạy cảm tại thị trường triển khai. SRS không tự tuyên bố tuân thủ chỉ bằng việc dùng encryption; cần legal/privacy review riêng.

---

## 9. Định hướng nghiên cứu AI

### 9.1 Pilot

Pilot dùng để chọn:

- construct/output chính;
- ground truth/reference;
- modality nằm trong primary analysis;
- mức burden nếu thử post-entry self-report;
- baseline, metric và success criterion.

Self-report không phải yêu cầu mặc định. Nếu nghiên cứu claim momentary affect, phải có validation subset với self-report phù hợp; nếu không, claim phải dừng ở sentiment, expressed emotion hoặc reflection fidelity.

### 9.2 Main study

Main study chỉ bắt đầu sau khi preregister hoặc khóa:

- population và sampling;
- inclusion/exclusion;
- primary question/estimand;
- output schema và reference;
- participant-aware split;
- metrics và minimum success criterion;
- missingness/error analysis;
- privacy, ethics và withdrawal protocol.

### 9.3 Product gate

Một output chỉ được chuyển từ Research sang Conditional/Released khi:

1. vượt baseline theo tiêu chí đã khóa;
2. có calibration/abstention phù hợp;
3. không có lỗi subgroup nghiêm trọng chưa xử lý;
4. presentation vượt UX comprehension/harm review;
5. privacy và license cho phép triển khai.

---

## 10. Ma trận truy vết tóm tắt

| Nhóm yêu cầu | Use Case | NFR chính | Thực thể chính |
|---|---|---|---|
| FR-C02–C05 Tạo entry | UC-01 | NFR-03, NFR-06–08, NFR-10 | JournalEntry, EntryVersion, ContentAsset |
| FR-C06–C07 Sửa/xóa/version | UC-03 | NFR-05, NFR-11 | EntryVersion, AIAnalysisResult |
| FR-C08 Offline/sync | UC-01 | NFR-06, NFR-09 | JournalEntry, ContentAsset |
| FR-C10–C12 Privacy/export | UC-03, UC-04 | NFR-01–05, NFR-17 | ConsentRecord, DataAccessEvent |
| FR-R01–R05 Pipeline/output | UC-02 | NFR-08–09, NFR-11, NFR-14–15 | AIProcessingJob, AIAnalysisResult |
| FR-R06–R09 Research governance | UC-02, UC-04 | NFR-12–13, NFR-16 | ResearchReference, DatasetVersion, ModelVersion, ModelEvaluation |
| FR-K01–K04 Product output | UC-05 | NFR-10–11, NFR-15–16 | AIAnalysisResult, PresentationRuleVersion |

Ma trận chi tiết một-một cho toàn bộ FR/NFR sẽ được duy trì trong công cụ quản lý yêu cầu hoặc phụ lục khi bước vào implementation.

---

## 11. Thuật ngữ

| Thuật ngữ | Giải thích |
|---|---|
| AIAnalysisResult | Output kỹ thuật có cấu trúc của model cho một entry/version |
| Annotation | Gán nhãn dữ liệu theo hướng dẫn |
| Baseline | Phương pháp cơ sở để so sánh |
| Calibration | Mức độ confidence phản ánh đúng xác suất thực |
| Committed | Yêu cầu sản phẩm đã chốt |
| Conditional | Chỉ phát hành khi vượt research/product gate |
| Construct | Khái niệm nghiên cứu muốn đo |
| Entry | Một bài/mục nhật ký |
| Expressed emotion | Cảm xúc được thể hiện trong nội dung |
| Ground truth/reference | Dữ liệu chuẩn/nguồn đối chiếu |
| Modality | Loại dữ liệu như text, voice, image hoặc video |
| Momentary affect | Affect gần thời điểm viết, cần self-report phù hợp nếu dùng làm target |
| Multimodal ablation | Thử bỏ từng modality để đo đóng góp |
| Participant-aware split | Không để cùng người xuất hiện ở cả train và test |
| Presentation layer | Lớp chuyển output kỹ thuật thành nội dung giao diện |
| Research | Yêu cầu chỉ dùng trong pilot/nghiên cứu |
| Self-report | Người tham gia tự đánh giá trạng thái của mình |
| Sentiment | Sắc thái của nội dung, không mặc định là cảm xúc thật |
| SDD | System Design Document — tài liệu thiết kế hệ thống |
| SRS | Software Requirements Specification — đặc tả yêu cầu phần mềm |

---

## 12. Điều kiện baseline tài liệu

SRS 1.3 chỉ được chuyển từ Draft sang Baselined khi nhóm và giảng viên thống nhất:

- phạm vi modality của MVP;
- trạng thái của video;
- output/reference dùng trong pilot;
- consent và privacy flow;
- success criterion của nghiên cứu;
- những Conditional features nào được phép triển khai thử;
- SDD đã giải quyết key management và processing architecture.

Cho đến thời điểm đó, output AI và cách hiển thị vẫn là giả thuyết nghiên cứu, không phải tính năng đã được đảm bảo.
