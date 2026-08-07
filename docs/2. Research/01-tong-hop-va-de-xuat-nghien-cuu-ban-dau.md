# Báo cáo tổng hợp quá trình hình thành hướng nghiên cứu

**Ngày cập nhật:** 2026-07-31  
**Mục đích:** Tóm tắt để báo cáo giảng viên về quá trình nhóm khảo sát sản phẩm, đọc nghiên cứu khoa học và lựa chọn hướng nghiên cứu cho ứng dụng nhật ký AI.

## Tóm tắt một trang

Nhóm muốn xây dựng một **ứng dụng nhật ký cá nhân**. Người dùng vẫn viết nhật ký như một app thông thường, có thể nhập chữ, giọng nói, hình ảnh hoặc video. AI là tính năng phân tích nội dung người dùng vừa tạo, không phải chatbot trị liệu.

Quá trình tìm hiểu được thực hiện theo ba bước:

1. **Khảo sát thị trường:** xem 11 ứng dụng nhật ký, mood tracking và AI journal để biết các sản phẩm hiện có đang giải quyết vấn đề như thế nào.
2. **Tổng hợp nghiên cứu khoa học:** đọc 12 nghiên cứu trực tiếp về diary/journal, 2 nghiên cứu liên quan và 1 nghiên cứu về cách chia dữ liệu.
3. **Thu hẹp đề tài:** chỉ giữ một bài toán ở cấp độ từng bài nhật ký, thay vì đồng thời nghiên cứu pattern dài hạn hoặc triệu chứng tâm lý.

Hướng nghiên cứu hiện tại là:

> Từ dữ liệu đa phương thức trong **một bài nhật ký**, AI có thể tạo ra dạng kết quả liên quan cảm xúc nào có thể được định nghĩa và đánh giá đáng tin cậy?

Dạng kết quả và cách tạo nhãn chuẩn vẫn chưa chốt. Self-report chỉ là một phương án sẽ được thử trong pilot, không phải bước bắt buộc của app chính thức.

---

## 1. Điểm xuất phát của nhóm

Sản phẩm được định nghĩa là **journal-first**:

- người dùng viết và lưu nhật ký như bình thường;
- AI đọc dữ liệu của bài vừa tạo và đưa ra một kết quả phù hợp;
- app không bắt người dùng khai báo cảm xúc trước khi viết;
- app không chẩn đoán bệnh và không thay thế chuyên gia tâm lý.

Ban đầu, nhóm cân nhắc nhiều hướng như nhận diện cảm xúc từng bài, tìm pattern qua nhiều bài và theo dõi thay đổi dài hạn. Sau quá trình tìm hiểu, nhóm quyết định chỉ nghiên cứu **một bài nhật ký tại một thời điểm** để phạm vi rõ ràng và có thể kiểm chứng.

---

## 2. Nhóm đã tìm hiểu các app hiện có như thế nào?

### 2.1 Phạm vi khảo sát

Nhóm thực hiện desk research trên 11 sản phẩm:

`Daylio`, `Day One`, `How We Feel`, `Stoic`, `Rosebud`, `MindDoc`, `Reflectly`, `Bearable`, `Apple Journal`, `Joplin` và `Reflection`.

Nguồn chủ yếu là website, help center, app store và privacy policy chính thức. Nhóm không xem đây là nghiên cứu người dùng trực tiếp và không kết luận rằng toàn bộ thị trường chỉ có 11 app này.

### 2.2 Các nhóm sản phẩm quan sát được

| Nhóm | Ví dụ | Cách hoạt động chính |
|---|---|---|
| Mood tracker | Daylio, Bearable | Người dùng tự chọn mood/activity rồi xem thống kê |
| Nhật ký truyền thống | Day One, Apple Journal | Viết tự do, lưu ảnh/audio/video, tìm kiếm và xem lại |
| Guided reflection | Stoic, Reflectly | Dùng prompt, routine và câu hỏi gợi ý |
| AI journal | Rosebud, Reflection | AI phản hồi bài viết, tìm lại lịch sử và tạo insight |
| Mental-health monitoring | MindDoc | Check-in có cấu trúc và báo cáo định kỳ |
| Privacy/local-first | Joplin | Ghi chú tự do, offline, export và quyền kiểm soát dữ liệu |

### 2.3 Những bài học chính từ thị trường

1. AI journal hiện nay thường tạo **reflection, câu hỏi và insight**, không chỉ trả về một nhãn tích cực/tiêu cực.
2. Mood do người dùng tự chọn vẫn là một cách làm phổ biến; AI không phải điều kiện bắt buộc để tạo thống kê.
3. Voice chỉ là một trong nhiều cách giảm công sức nhập liệu; prompt, template và suggestion cũng có vai trò tương tự.
4. Privacy, export, quyền xóa dữ liệu và quyền tắt AI là phần quan trọng của sản phẩm.
5. Một số app có giao diện tiếng Việt, nhưng chưa thấy benchmark công khai đủ mạnh về việc AI hiểu nhật ký tiếng Việt, teencode hoặc giọng vùng miền.
6. Các tính năng nhóm nghĩ đến phần lớn đã tồn tại ở đâu đó. Cơ hội của nhóm không nằm ở một tính năng “chưa ai có”, mà ở cách kết hợp phù hợp với người dùng Việt Nam và có kiểm chứng khoa học.

Từ khảo sát này, nhóm dùng các sản phẩm như ví dụ để hiểu không gian giải pháp; nhóm không xem nội dung marketing của sản phẩm là bằng chứng khoa học về hiệu quả.

---

## 3. Nhóm đã đọc các nghiên cứu khoa học như thế nào?

### 3.1 Phạm vi review

Nhóm thực hiện systematic scoping review có audit trail, gồm:

- 12 nghiên cứu trực tiếp dùng diary/journal;
- 2 nghiên cứu liên quan để kiểm tra ảnh hưởng của prompt và domain;
- 1 nghiên cứu phương pháp về nguy cơ rò rỉ dữ liệu giữa train và test.

Phần lớn nghiên cứu dùng tiếng Anh, mẫu phương Tây hoặc nhóm lâm sàng đặc thù. Nhóm chưa tìm thấy một nghiên cứu validation độc lập trên nhật ký cá nhân tiếng Việt có thiết kế tương đương.

### 3.2 Điều quan trọng nhất: phải phân biệt đúng khái niệm

| Khái niệm | Nghĩa đơn giản | Cách kiểm chứng phù hợp |
|---|---|---|
| Sentiment | Sắc thái tích cực, tiêu cực hoặc trung tính của nội dung | Người gán nhãn hoặc dataset sentiment |
| Expressed emotion | Cảm xúc được thể hiện trong bài | Người gán nhãn theo hướng dẫn rõ ràng |
| Momentary affect | Trạng thái cảm xúc người viết trải nghiệm lúc đó | Self-report gần thời điểm viết |
| Symptom | Mức độ triệu chứng trong một khoảng thời gian | Thang đo tâm lý đã được kiểm định |
| Diagnosis | Chẩn đoán lâm sàng | Đánh giá/phỏng vấn lâm sàng chuẩn hóa |

Sentiment của câu chữ không tự động bằng cảm xúc thật của người viết. Vì vậy tên output, loại nhãn và tuyên bố của nghiên cứu phải khớp nhau.

### 3.3 Một số nghiên cứu ảnh hưởng trực tiếp đến quyết định của nhóm

| Nghiên cứu | Kết quả chính | Bài học cho nhóm |
|---|---|---|
| [Tov et al. (2013)](https://doi.org/10.1037/a0033007) | Từ ngữ tiêu cực liên hệ với negative affect tự báo cáo, nhưng tín hiệu tích cực không ổn định | Text có tín hiệu, nhưng không phải chiều cảm xúc nào cũng dễ nhận diện |
| [Chen & Golab (2020)](https://doi.org/10.1007/s00607-019-00777-6) | Dataset rất lớn nhưng mood classification chỉ khoảng 33% | Nhiều dữ liệu không tự động bảo đảm model chính xác |
| [Linton et al. (2021)](https://doi.org/10.2196/25279) | Nội dung prompt làm thay đổi rõ tone của nhật ký | Prompt là một phần của phép đo, không chỉ là nội dung giao diện |
| [Collins et al. (2025)](https://doi.org/10.1037/abn0001003) | Dự đoán symptom theo tuần chỉ đạt mức vừa phải | Không nên suy từ text thành công cụ sàng lọc/chẩn đoán độc lập |
| [Fisher et al. (2026)](https://doi.org/10.1037/abn0001150) | Hiệu năng khác nhau lớn giữa từng người | Cần phân tích lỗi theo người, không chỉ báo metric trung bình |
| [CALLM (2025)](https://doi.org/10.48550/arXiv.2503.10707) | LLM có context đạt khoảng 73% balanced accuracy cho affect đồng thời | Kết quả đáng chú ý nhưng là preprint và population rất đặc thù |
| [Karbalaie et al. (2026)](https://doi.org/10.2196/87728) | Accuracy giảm mạnh khi tách train/test đúng theo participant | Không được để bài của cùng một người ở cả train và test |

### 3.4 Kết luận rút ra từ khoa học

1. Nhật ký có chứa tín hiệu liên quan cảm xúc, nhưng độ mạnh thường chỉ ở mức vừa phải.
2. Không có một model hoặc ontology cảm xúc nào đã được chứng minh là tốt nhất cho mọi bối cảnh.
3. Dataset social media tiếng Việt có thể dùng để khởi tạo model, nhưng không thay thế dữ liệu nhật ký thật.
4. Ground truth phải phù hợp với điều nhóm muốn tuyên bố.
5. Cần baseline đơn giản, tập test chưa từng thấy, calibration và phân tích lỗi.
6. Chưa có đủ cơ sở để nghiên cứu chẩn đoán, self-harm hoặc symptom tracking trong phạm vi hiện tại.

Các kết luận trên được rút ra từ việc đọc phần phương pháp, kết quả và giới hạn của từng nghiên cứu, không chỉ từ abstract hoặc số accuracy được công bố.

---

## 4. Nhóm đã thu hẹp đề tài như thế nào?

```text
Khảo sát app hiện có
        +
Đọc nghiên cứu khoa học
        ↓
Nhận ra phạm vi ban đầu quá rộng
        ↓
Loại nghiên cứu pattern dài hạn và symptom tracking
        ↓
Giữ một bài toán: phân tích một entry đa phương thức
        ↓
Pilot để chọn output và ground truth phù hợp
```

Quyết định này giúp:

- mỗi mẫu nghiên cứu có đơn vị rõ ràng là một entry;
- so sánh được đóng góp của text, voice, image và video;
- tránh trộn emotion, mood, symptom và diagnosis;
- giữ phạm vi khả thi cho nhóm;
- liên kết trực tiếp kết quả nghiên cứu với tính năng app.

---

## 5. Hướng nghiên cứu hiện tại

### 5.1 Tên đề tài tạm thời

**Tiếng Việt**

> **Phân tích liên quan cảm xúc ở cấp độ bài viết từ nhật ký cá nhân đa phương thức tiếng Việt: Nghiên cứu và đánh giá các dạng đầu ra AI**

**Tiếng Anh**

> **Entry-Level Emotion-Related Analysis of Multimodal Vietnamese Personal Journals: Investigating and Evaluating AI Outputs**

Tên này là tên tạm thời vì output và ground truth chưa được chốt.

### 5.2 Câu hỏi nghiên cứu

> Từ dữ liệu đa phương thức trong một entry nhật ký cá nhân tiếng Việt, AI có thể tạo ra dạng output liên quan cảm xúc nào có thể được định nghĩa và đánh giá đáng tin cậy?

Các câu hỏi phụ:

- Mỗi modality đóng góp bao nhiêu vào kết quả?
- Kết hợp nhiều modality có tốt hơn chỉ dùng text không?
- Output nào vừa đo được trong nghiên cứu, vừa dễ hiểu và an toàn trong app?

### 5.3 Một luồng nghiên cứu duy nhất

```text
Người dùng tạo một entry
  ├─ Text
  ├─ Voice
  ├─ Image
  └─ Video (nếu khả thi)
        ↓
AI chuyển dữ liệu thành tín hiệu có thể phân tích
        ↓
AI tạo candidate output cho chính entry đó
        ↓
Pilot đánh giá bằng reference phù hợp
        ↓
Chọn một output và ground truth cho nghiên cứu chính
        ↓
Đánh giá model trên dữ liệu chưa từng thấy
```

### 5.4 Các output đang được cân nhắc

| Output | Nhóm đang đo điều gì? | Reference phù hợp |
|---|---|---|
| Sentiment | Sắc thái của nội dung | Người gán nhãn/dataset phù hợp |
| Expressed emotion | Cảm xúc được biểu hiện trong entry | Người gán nhãn theo codebook |
| Valence/arousal | Dễ chịu–khó chịu và bình lặng–kích hoạt | Self-report hoặc annotation tùy định nghĩa |
| Momentary affect | Cảm xúc người dùng trải nghiệm lúc viết | Self-report gần thời điểm |
| Reflection định tính | AI tóm tắt/diễn giải entry có trung thành và hữu ích không | Người dùng đánh giá + kiểm tra factuality |

### 5.5 Vai trò của self-report

Self-report **chưa được chốt là thành phần bắt buộc của methodology**.

Nhóm có thể thử self-report sau một số entry trong pilot để kiểm tra:

- người dùng có thấy phiền không;
- câu hỏi có làm thay đổi trải nghiệm viết không;
- dữ liệu có đủ ổn định để chọn momentary affect làm target không.

Nếu nghiên cứu cuối cùng tuyên bố ước lượng cảm xúc người dùng trải nghiệm, cần một validation subset có self-report. Nếu không dùng self-report, output phải được gọi là sentiment, expressed emotion hoặc reflection, không phải cảm xúc thật.

App chính thức không bắt người dùng self-report trước hoặc sau mỗi bài viết.

### 5.6 Cách đánh giá dự kiến

- So sánh baseline đơn giản với model phức tạp hơn.
- Tách train/test theo participant nếu mỗi người có nhiều entry.
- So sánh text-only, voice transcript, acoustic, image và multimodal fusion.
- Báo Macro-F1/balanced accuracy cho classification hoặc MAE/RMSE cho score liên tục.
- Kiểm tra calibration, trường hợp model từ chối dự đoán và các lỗi trên teencode, code-switching, ASR, giọng vùng miền và ảnh không liên quan.

Cỡ mẫu, thời lượng pilot và output chính sẽ được khóa sau bước feasibility, không quyết định chỉ dựa trên benchmark công khai.

---

## 6. Quan hệ giữa nghiên cứu và tính năng app

```text
Nghiên cứu chứng minh được điều gì
                ↓
Chỉ triển khai đúng loại output đó
                ↓
Hiển thị uncertainty và quyền từ chối/sửa
                ↓
Theo dõi lỗi sau triển khai
```

Ví dụ:

- Nếu chỉ validation được sentiment, app chỉ mô tả sắc thái nội dung.
- Nếu có self-report và validation tốt, app mới có thể nói về momentary affect.
- Nếu reflection được đánh giá tốt hơn classification, app có thể ưu tiên phản hồi định tính.
- Nếu multimodal không tốt hơn text, nhóm không cần triển khai hệ thống fusion phức tạp.

### 6.1 Tách output kỹ thuật và cách biểu hiện trong giao diện

Nghiên cứu AI và thiết kế giao diện là hai bước nối tiếp nhưng khác nhau:

```text
Output kỹ thuật của model
  (nhãn, score, confidence, evidence)
                ↓
Quy tắc chuyển đổi sang nội dung cho người dùng
                ↓
Cách biểu hiện trong giao diện
  (câu reflection, thẻ cảm xúc, màu sắc, biểu đồ hoặc câu hỏi xác nhận)
                ↓
Nghiên cứu UX để kiểm tra người dùng có hiểu đúng không
```

Một output kỹ thuật không tương ứng với duy nhất một cách hiển thị. Ví dụ, cùng một score có thể được biểu hiện bằng thanh mức độ, một câu mô tả hoặc không hiển thị nếu confidence thấp.

Sau khi nghiên cứu AI xác định output nào đáng tin cậy, nhóm mới thiết kế và thử nghiệm nhiều cách biểu hiện. Tiêu chí lựa chọn gồm: dễ hiểu, không gây tự chẩn đoán, thể hiện được uncertainty và cho phép người dùng xác nhận hoặc từ chối.

---

## 7. Những gì đã chốt và chưa chốt

### Đã chốt

- Sản phẩm là app nhật ký, không phải chatbot trị liệu.
- Nghiên cứu chỉ xử lý một entry tại một thời điểm.
- Input có thể gồm text, voice, image và video.
- Không nghiên cứu pattern dài hạn, symptom tracking hoặc personalization ở giai đoạn này.
- Không hỏi cảm xúc trước khi viết.
- Self-report không phải bước bắt buộc của app.
- Không chẩn đoán hoặc suy nguyên nhân từ output.

### Chưa chốt

- Output chính và tên đề tài cuối cùng.
- Ground truth/reference.
- Có dùng self-report trong pilot hay không và trên bao nhiêu entry.
- Video có nằm trong pilot đầu không.
- Population, cỡ mẫu và thời gian thu thập.
- Model, fusion strategy và tiêu chí thành công.

---

## 8. Ranh giới an toàn và đạo đức

- Không gọi sentiment là cảm xúc thật.
- Không gọi output của một entry là symptom score hoặc diagnosis.
- Không suy cảm xúc bên trong chỉ từ khuôn mặt hoặc giọng nói.
- Dữ liệu text, audio, ảnh, video, transcript và embedding đều là dữ liệu nhạy cảm.
- Consent phải nói rõ dữ liệu nào được lưu, gửi cho bên thứ ba hoặc dùng để huấn luyện.
- Người dùng cần quyền tắt AI, đánh dấu entry riêng tư, xóa và export dữ liệu.
- Crisis/self-harm detection cần một nghiên cứu và quy trình chuyên môn riêng.

---

## 9. Kết luận

Nhóm không bắt đầu bằng việc chọn sẵn PhoBERT, LLM hay một bộ nhãn cảm xúc rồi tìm cách chứng minh model đúng. Nhóm bắt đầu từ khảo sát sản phẩm, kiểm tra bằng chứng khoa học, phân biệt các construct và thu hẹp thành một câu hỏi có thể đánh giá.

Kết quả mong muốn của pilot không nhất thiết là một model hoàn chỉnh. Thành công trước tiên là xác định được:

1. output nào có ý nghĩa;
2. ground truth nào phù hợp và khả thi;
3. modality nào thật sự tạo thêm giá trị;
4. claim nào đủ an toàn để chuyển thành tính năng app.

---

## Phụ lục A — Từ khóa Anh–Việt

| English | Tiếng Việt dễ hiểu |
|---|---|
| Affect | Trạng thái cảm xúc nói chung |
| Annotation | Gán nhãn dữ liệu |
| Baseline | Phương pháp cơ sở để so sánh |
| Calibration | Mức độ điểm tin cậy phản ánh đúng xác suất thực |
| Classification | Phân loại |
| Codebook | Bộ hướng dẫn và định nghĩa dùng để gán nhãn |
| Construct | Khái niệm/đặc tính mà nghiên cứu muốn đo |
| Dataset | Tập dữ liệu |
| Desk research | Nghiên cứu tài liệu có sẵn |
| Diagnosis | Chẩn đoán |
| Emotion recognition | Nhận diện cảm xúc theo một bộ nhãn xác định |
| Entry | Một bài/mục nhật ký |
| Expressed emotion | Cảm xúc được thể hiện trong nội dung |
| Ground truth | Dữ liệu chuẩn dùng để đối chiếu |
| Longitudinal study | Nghiên cứu theo dõi cùng đối tượng qua thời gian |
| Modality | Loại dữ liệu đầu vào, như text, voice hoặc image |
| Momentary affect | Trạng thái cảm xúc gần thời điểm hiện tại |
| Multimodal | Kết hợp nhiều loại dữ liệu |
| Ontology | Hệ thống các khái niệm/nhãn được định nghĩa |
| Participant-aware split | Chia dữ liệu sao cho cùng người không nằm ở cả train và test |
| Pilot study | Nghiên cứu thử nghiệm quy mô nhỏ |
| Reflection | Phản hồi giúp người dùng nhìn lại nội dung vừa viết |
| Self-report | Người tham gia tự đánh giá trạng thái của mình |
| Sentiment analysis | Phân tích sắc thái tích cực/tiêu cực/trung tính của nội dung |
| Symptom | Triệu chứng |
| Validation | Kiểm chứng/đánh giá trên dữ liệu phù hợp |
| Valence | Mức dễ chịu–khó chịu |
| Arousal | Mức bình lặng–kích hoạt/căng thẳng |

---

## Phụ lục B — Nguồn trực tiếp

### Nguồn sản phẩm chính

- [Daylio](https://daylio.net/)
- [Day One](https://dayoneapp.com/)
- [How We Feel](https://howwefeel.org/)
- [Stoic](https://www.getstoic.com/features)
- [Rosebud](https://www.rosebud.app/)
- [MindDoc](https://minddoc.com/de/en)
- [Reflectly](https://apps.apple.com/us/app/reflectly-journal-ai-diary/id1241229134)
- [Bearable](https://bearable.app/)
- [Apple Journal](https://apps.apple.com/us/app/journal/id6447391597)
- [Joplin](https://joplinapp.org/help/)
- [Reflection](https://www.reflection.app/ai)

Các trang sản phẩm chỉ chứng minh nhà phát triển công bố tính năng gì; chúng không chứng minh hiệu quả tâm lý hoặc độ chính xác của AI.

### Nguồn khoa học chính

- [Tov et al. (2013)](https://doi.org/10.1037/a0033007)
- [Chen & Golab (2020)](https://doi.org/10.1007/s00607-019-00777-6)
- [Linton et al. (2021)](https://doi.org/10.2196/25279)
- [Collins et al. (2025)](https://doi.org/10.1037/abn0001003)
- [Fisher et al. (2026)](https://doi.org/10.1037/abn0001150)
- [CALLM (2025)](https://doi.org/10.48550/arXiv.2503.10707)
- [Karbalaie et al. (2026)](https://doi.org/10.2196/87728)
