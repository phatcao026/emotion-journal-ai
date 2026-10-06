"""
Emotion Journal AI - Multimodal Annotation Studio
Ứng dụng Mini App hỗ trợ gán nhãn dữ liệu Đa phương thức (Text + Audio)
Đề tài: Multimodal Emotion Recognition for Vietnamese Language Diary
"""

import os
import sys
import streamlit as st
import pandas as pd
import json

# Thêm đường dẫn project vào sys.path để import an toàn
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.annotation_tool.annotation_manager import AnnotationManager

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Emotion Journal AI - Annotation Studio",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, trực quan
st.markdown("""
<style>
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.85rem;
        color: #888888;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .badge-joy { background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 2px 8px; font-weight: 600; font-size: 0.8rem; }
    .badge-sadness { background-color: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 6px; padding: 2px 8px; font-weight: 600; font-size: 0.8rem; }
    .badge-anxiety { background-color: rgba(168, 85, 247, 0.15); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 6px; padding: 2px 8px; font-weight: 600; font-size: 0.8rem; }
    .badge-anger { background-color: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 6px; padding: 2px 8px; font-weight: 600; font-size: 0.8rem; }
    .badge-neutral { background-color: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 6px; padding: 2px 8px; font-weight: 600; font-size: 0.8rem; }
    .text-box {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        font-size: 0.95rem;
        line-height: 1.6;
        max-height: 380px;
        overflow-y: auto;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Khởi tạo AnnotationManager & Session State
# ---------------------------------------------------------
@st.cache_resource
def get_manager():
    return AnnotationManager(
        crawled_meta_path=os.path.join(BASE_DIR, "data", "crawled", "metadata.jsonl"),
        annotation_dir=os.path.join(BASE_DIR, "data", "annotated"),
        taxonomy_path=os.path.join(BASE_DIR, "data", "taxonomy.json")
    )

manager = get_manager()

if "sample_index" not in st.session_state:
    st.session_state.sample_index = 0
if "annotator_id" not in st.session_state:
    st.session_state.annotator_id = "annotator_01"
if "auto_filled" not in st.session_state:
    st.session_state.auto_filled = {}

# ---------------------------------------------------------
# Sidebar: Bộ lọc, Thống kê & Quản lý phiên
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/mental-health.png", width=50)
    st.markdown("### 🎙️ Emotion Studio")
    st.caption("Multimodal Emotion Annotation v1.0")

    # 1. Định danh Annotator
    annotator_input = st.text_input(
        "Mã Người Gán Nhãn (Annotator ID):",
        value=st.session_state.annotator_id,
        help="Định danh này giúp đo lường chỉ số đồng thuận liên quan sát viên (Cohen's Kappa / Fleiss' Kappa) giữa nhiều thành viên."
    )
    if annotator_input != st.session_state.annotator_id:
        st.session_state.annotator_id = annotator_input
        st.rerun()

    st.divider()

    # 2. Bộ lọc danh sách mẫu
    st.markdown("#### 🔍 Bộ Lọc Danh Sách")
    modality_choice = st.selectbox(
        "Phương thức (Modality):",
        ["Tất cả (All)", "📝 Chỉ Text", "🎙️ Chỉ Voice", "🔗 Chỉ Text+Voice", "🚨 Nguy cơ Crisis"],
        index=0
    )
    mod_map = {
        "Tất cả (All)": "all",
        "📝 Chỉ Text": "text",
        "🎙️ Chỉ Voice": "voice",
        "🔗 Chỉ Text+Voice": "text_voice",
        "🚨 Nguy cơ Crisis": "crisis"
    }

    status_choice = st.radio(
        "Trạng thái gán nhãn:",
        ["Tất cả", "Chưa gán nhãn", "Đã gán nhãn"],
        horizontal=True
    )
    status_map = {
        "Tất cả": "all",
        "Chưa gán nhãn": "unannotated",
        "Đã gán nhãn": "annotated"
    }

    search_query = st.text_input("Tìm kiếm (ID, nội dung, tác giả):", value="")

    # Lấy danh sách mẫu thỏa mãn bộ lọc
    filtered_samples = manager.get_filtered_samples(
        modality_filter=mod_map[modality_choice],
        status_filter=status_map[status_choice],
        annotator_id=st.session_state.annotator_id,
        search_query=search_query
    )

    st.divider()

    # 3. Thống kê tiến độ
    stats = manager.get_statistics(annotator_id=st.session_state.annotator_id)
    st.markdown("#### 📊 Tiến Độ Của Bạn")
    st.progress(stats["percentage_done"] / 100.0)
    st.caption(f"Đã gán: **{stats['total_annotated']}** / {stats['total_samples']} mẫu ({stats['percentage_done']}%)")

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Valence TB", f"{stats['avg_valence']}")
    with col_stat2:
        st.metric("Arousal TB", f"{stats['avg_arousal']}")

    if stats["crisis_count"] > 0:
        st.error(f"🚨 Phát hiện {stats['crisis_count']} mẫu rủi ro tự hại!")

    # Biểu đồ phân bố nhãn đã gán
    if stats["primary_distribution"]:
        st.markdown("**Phân bố cảm xúc đã gán:**")
        df_dist = pd.DataFrame(list(stats["primary_distribution"].items()), columns=["Emotion", "Count"])
        st.bar_chart(df_dist.set_index("Emotion"), height=160)

    # Nút xuất file
    st.divider()
    if os.path.exists(manager.annotations_csv):
        with open(manager.annotations_csv, "r", encoding="utf-8") as f:
            csv_data = f.read()
        st.download_button(
            label="📥 Tải Bảng Nhãn CSV",
            data=csv_data,
            file_name="emotion_annotations.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---------------------------------------------------------
# Main Panel: Workspace Gán Nhãn
# ---------------------------------------------------------
st.markdown('<div class="main-header">🏷️ Multimodal Emotion Annotation Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Nền tảng gán nhãn đa phương thức chuyên biệt cho Nhật ký & Tâm sự tiếng Việt (Hierarchical Taxonomy + Continuous VAD + Crisis Triage)</div>', unsafe_allow_html=True)

if not filtered_samples:
    st.warning("⚠️ Không tìm thấy mẫu nào phù hợp với bộ lọc hiện tại. Vui lòng chuyển bộ lọc trong Sidebar.")
    st.stop()

# Đảm bảo index hợp lệ
if st.session_state.sample_index >= len(filtered_samples):
    st.session_state.sample_index = 0
current_sample = filtered_samples[st.session_state.sample_index]
current_id = current_sample.get("sample_id", "")

# ---------------------------------------------------------
# Thanh Điều Hướng Trên Cùng
# ---------------------------------------------------------
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1.5, 3, 1.5, 2])

with nav_col1:
    if st.button("⬅️ Mẫu trước (Prev)", use_container_width=True, disabled=(st.session_state.sample_index == 0)):
        st.session_state.sample_index -= 1
        st.rerun()

with nav_col2:
    selected_idx = st.selectbox(
        "Nhảy đến mẫu:",
        range(len(filtered_samples)),
        index=st.session_state.sample_index,
        format_func=lambda i: f"#{i+1}: {filtered_samples[i].get('sample_id')} [{filtered_samples[i].get('modality')}]",
        label_visibility="collapsed"
    )
    if selected_idx != st.session_state.sample_index:
        st.session_state.sample_index = selected_idx
        st.rerun()

with nav_col3:
    if st.button("Mẫu sau (Next) ➡️", use_container_width=True, disabled=(st.session_state.sample_index >= len(filtered_samples) - 1)):
        st.session_state.sample_index += 1
        st.rerun()

with nav_col4:
    st.write(f"**Vị trí: {st.session_state.sample_index + 1} / {len(filtered_samples)}**")

st.write("")

# ---------------------------------------------------------
# Lấy nhãn hiện có (nếu đã từng gán) hoặc nhãn gợi ý (Model-Assisted)
# ---------------------------------------------------------
existing_ann = manager.get_existing_annotation(current_id, st.session_state.annotator_id)
pseudo_primary = current_sample.get("primary_label", "JOY")
pseudo_sub = current_sample.get("sub_label", "CALM")
pseudo_score = current_sample.get("emotion_score", 0.8)

# Xác định giá trị mặc định cho form
default_primary = existing_ann.get("primary_label") if existing_ann else pseudo_primary
default_sub = existing_ann.get("sub_label") if existing_ann else pseudo_sub
default_valence = float(existing_ann.get("valence", 0.0)) if existing_ann else (0.4 if default_primary == "JOY" else (-0.4 if default_primary in ["SADNESS", "ANXIETY", "ANGER"] else 0.0))
default_arousal = float(existing_ann.get("arousal", 0.3)) if existing_ann else (0.7 if default_primary == "ANGER" else 0.3)
default_congruence = existing_ann.get("congruence", "congruent") if existing_ann else "congruent"
default_crisis = existing_ann.get("crisis_level", 0) if existing_ann else (3 if current_sample.get("is_crisis") else 0)
default_notes = existing_ann.get("notes", "") if existing_ann else ""

# ---------------------------------------------------------
# Bố Cục 2 Cột: Bên Trái (Dữ Liệu) - Bên Phải (Gán Nhãn)
# ---------------------------------------------------------
col_data, col_label = st.columns([1.1, 0.9], gap="large")

# ==================== CỘT TRÁI: DỮ LIỆU ĐA PHƯƠNG THỨC ====================
with col_data:
    # 1. Thẻ Metadata tóm tắt
    mod = current_sample.get("modality", "text")
    is_crisis = current_sample.get("is_crisis", False) or mod == "crisis_quarantine"
    
    st.markdown(f"#### 📄 Mẫu: `{current_id}`")
    
    meta_badges = f"""
    <span class="badge-neutral">Phương thức: <b>{mod.upper()}</b></span> &nbsp;
    <span class="badge-neutral">Nguồn: <b>{current_sample.get('source_platform', 'unknown')}</b></span> &nbsp;
    <span class="badge-neutral">Tác giả ID: <b>{current_sample.get('author_hash', 'anon')[:8]}</b></span>
    """
    st.markdown(meta_badges, unsafe_allow_html=True)

    # Cảnh báo cách ly nếu có
    if is_crisis or current_sample.get("crisis_keywords"):
        st.error(f"🚨 **CẢNH BÁO KHỦNG HOẢNG TÂM LÝ:** Phát hiện từ khóa nhạy cảm: `{current_sample.get('crisis_keywords', ['nhảy lầu / tự hại'])}`. Vui lòng xem xét cẩn trọng mục Tầm soát Khủng hoảng.")

    st.write("")

    # 2. Hiển thị Âm thanh (Audio Player)
    audio_path = manager.get_audio_path(current_sample)
    if audio_path and os.path.exists(audio_path):
        st.markdown("##### 🎙️ Bản Ghi Âm Thanh (16kHz Mono)")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")

        # Thẻ thông số âm học
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        with col_a1:
            st.metric("Thời lượng", f"{current_sample.get('duration_s', 0):.1f}s")
        with col_a2:
            st.metric("RMS Energy", f"{current_sample.get('rms_dbfs', 0):.1f} dB")
        with col_a3:
            st.metric("Speech Ratio", f"{int(current_sample.get('speech_ratio', 0)*100)}%")
        with col_a4:
            st.metric("Peak Headroom", f"{current_sample.get('peak', 0):.3f}")

    # 3. Hiển thị Văn bản (Text Content)
    text_content = current_sample.get("text_content") or current_sample.get("text") or ""
    if text_content:
        st.markdown("##### 📝 Nội Dung Nhật Ký / Tâm Sự")
        if current_sample.get("title"):
            st.markdown(f"**Tiêu đề:** *{current_sample.get('title')}*")
        
        st.markdown(f'<div class="text-box">{text_content}</div>', unsafe_allow_html=True)

        # Các chỉ dấu tự sự & từ cảm xúc phát hiện
        pronouns = current_sample.get("matched_pronouns", [])
        keywords = current_sample.get("matched_keywords", [])
        
        if pronouns or keywords:
            st.write("")
            col_p, col_k = st.columns(2)
            with col_p:
                st.caption(f"**Đại từ ngôi thứ nhất:** {', '.join(pronouns[:6]) if pronouns else 'None'}")
            with col_k:
                st.caption(f"**Từ khóa cảm xúc bắt được:** {', '.join(keywords[:6]) if keywords else 'None'}")

    if not audio_path and not text_content:
        st.info("Mẫu này không chứa tệp nội dung trực tiếp.")

# ==================== CỘT PHẢI: BẢNG ĐIỀU KHIỂN GÁN NHÃN ====================
with col_label:
    st.markdown("#### 🎯 Không Gian Gán Nhãn (Annotation Workspace)")

    # 1. Hỗ trợ Model-Assisted (Gợi ý nhãn sơ bộ)
    status_icon = "✅ ĐÃ GÁN NHÃN" if existing_ann else "⏳ CHƯA GÁN NHÃN"
    status_color = "green" if existing_ann else "orange"
    st.markdown(f"Trạng thái: :{status_color}[**{status_icon}**]")

    col_sugg1, col_sugg2 = st.columns([2, 1])
    with col_sugg1:
        st.info(f"🤖 **Gợi ý sơ bộ:** `{pseudo_primary}` / `{pseudo_sub}` (Tin cậy: {int(pseudo_score*100)}%)")
    with col_sugg2:
        if st.button("⚡ Áp dụng Gợi ý", use_container_width=True):
            st.session_state.auto_filled[current_id] = True
            st.toast("Đã điền nhanh nhãn gợi ý!", icon="⚡")
            st.rerun()

    # Form Gán Nhãn Chính
    with st.form(key=f"label_form_{current_id}"):
        # A. Cảm xúc Chính (Primary 5 classes)
        primary_list = ["JOY", "SADNESS", "ANXIETY", "ANGER", "NEUTRAL"]
        primary_idx = primary_list.index(default_primary) if default_primary in primary_list else 0

        selected_primary = st.radio(
            "1. Nhóm Cảm Xúc Chính (Primary Emotion):",
            primary_list,
            index=primary_idx,
            horizontal=True,
            help="JOY (Niềm vui, An yên, Hy vọng), SADNESS (Nỗi buồn, Cô đơn, Mặc cảm), ANXIETY (Lo âu, Bất an, Sợ hãi), ANGER (Tức giận, Phẫn nộ, Kinh tởm), NEUTRAL (Bình thản, Chiêm nghiệm)."
        )

        # B. Cảm xúc Chi tiết (Sub-emotions 11 classes liên kết tự động)
        # Lọc danh sách sub-emotions theo primary đã chọn
        sub_to_primary = manager.taxonomy.get("sub_to_primary_mapping", {})
        allowed_subs = [s for s, p in sub_to_primary.items() if p == selected_primary]
        if selected_primary == "NEUTRAL":
            allowed_subs = ["CALM", "NEUTRAL"]

        sub_idx = allowed_subs.index(default_sub) if default_sub in allowed_subs else 0
        selected_sub = st.selectbox(
            "2. Nhãn Cảm Xúc Chi Tiết (Fine-grained Sub-emotion):",
            allowed_subs,
            index=sub_idx,
            help="Các nhãn cảm xúc thứ cấp được tự động khớp theo Taxonomy chuẩn của đề tài."
        )

        st.divider()

        # C. Chiều Không Gian Cảm Xúc Liên Tục (Dimensional VAD)
        col_v, col_a = st.columns(2)
        with col_v:
            selected_valence = st.slider(
                "3. Valence (Mức độ dễ chịu):",
                min_value=-1.0,
                max_value=1.0,
                value=float(default_valence),
                step=0.05,
                help="-1.0 (Cực kỳ tiêu cực/Đau khổ) ➔ 0.0 (Trung tính) ➔ +1.0 (Cực kỳ tích cực/Hạnh phúc)"
            )
        with col_a:
            selected_arousal = st.slider(
                "4. Arousal (Mức độ kích hoạt):",
                min_value=0.0,
                max_value=1.0,
                value=float(default_arousal),
                step=0.05,
                help="0.0 (Trầm lắng, thì thầm, u uất) ➔ 1.0 (Kích động, gào thét, cuồng nhiệt)"
            )

        # D. Xung đột Đa phương thức (Multimodal Congruence)
        congruence_options = [
            ("congruent", "🟢 Nhất quán (Lời & Giọng đồng nhất)"),
            ("masking", "🎭 Che giấu cảm xúc (Lời nói ổn nhưng giọng buồn/run rẩy)"),
            ("sarcasm", "😏 Châm biếm / Mỉa mai (Sarcasm)"),
            ("na", "⚪ Không áp dụng (Chỉ 1 phương thức đơn lẻ)")
        ]
        cong_vals = [c[0] for c in congruence_options]
        cong_idx = cong_vals.index(default_congruence) if default_congruence in cong_vals else 0
        
        selected_congruence = st.selectbox(
            "5. Tính Nhất Quán Đa Phương Thức (Multimodal Congruence):",
            cong_vals,
            index=cong_idx,
            format_func=lambda x: dict(congruence_options)[x],
            help="Đánh giá xem cảm xúc từ lời văn bản có khớp với âm sắc giọng nói hay có sự kìm nén/che giấu cảm xúc."
        )

        # E. Tầm Soát Nguy Cơ Khủng Hoảng (Crisis Triage)
        crisis_options = [
            (0, "0: An toàn (None - Tâm sự đời thường)"),
            (1, "1: Căng thẳng thường nhật (Mild Distress - Stress học tập/công việc)"),
            (2, "2: Tuyệt vọng kéo dài (Severe Distress - Bế tắc, kiệt quệ tâm lý)"),
            (3, "3: Nguy cơ tự hại / Cấp cứu (Acute Crisis - Ý định tự tử, tự làm đau bản thân)")
        ]
        crisis_vals = [c[0] for c in crisis_options]
        crisis_idx = crisis_vals.index(default_crisis) if default_crisis in crisis_vals else 0

        selected_crisis = st.radio(
            "6. Tầm Soát Rủi Ro Tâm Lý (Crisis Triage):",
            crisis_vals,
            index=crisis_idx,
            format_func=lambda x: dict(crisis_options)[x],
            help="Quy chuẩn tầm soát an toàn theo hướng dẫn y tế Columbia-SSRS. Mức 3 sẽ tự động chuyển vào khu vực cách ly an toàn."
        )

        # F. Ghi Chú Của Annotator
        annotator_notes = st.text_input(
            "7. Ghi chú cá nhân (Notes / Phương ngữ / Tạp âm):",
            value=default_notes,
            placeholder="Ví dụ: Giọng miền Trung nhẹ, có tiếng xe cộ nền, từ ngữ ẩn dụ sâu..."
        )

        st.divider()

        # Nút Hành Động
        btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
        with btn_col1:
            submit_save = st.form_submit_button("💾 LƯU & TIẾP TỤC (Save & Next)", use_container_width=True, type="primary")
        with btn_col2:
            submit_skip = st.form_submit_button("⏭️ Bỏ Qua (Skip)", use_container_width=True)
        with btn_col3:
            submit_quarantine = st.form_submit_button("🚨 Cách Ly (Quarantine)", use_container_width=True)

    # Xử lý sự kiện Submit
    if submit_save:
        manager.save_annotation(
            sample_id=current_id,
            annotator_id=st.session_state.annotator_id,
            primary_label=selected_primary,
            sub_label=selected_sub,
            valence=selected_valence,
            arousal=selected_arousal,
            congruence=selected_congruence,
            crisis_level=selected_crisis,
            notes=annotator_notes,
            action_type="annotated"
        )
        st.toast(f"Đã lưu nhãn mẫu #{st.session_state.sample_index + 1} ({selected_primary} / {selected_sub})!", icon="✅")
        if st.session_state.sample_index < len(filtered_samples) - 1:
            st.session_state.sample_index += 1
        st.rerun()

    if submit_skip:
        manager.save_annotation(
            sample_id=current_id,
            annotator_id=st.session_state.annotator_id,
            primary_label="UNKNOWN",
            sub_label="UNKNOWN",
            valence=0.0,
            arousal=0.0,
            congruence="na",
            crisis_level=0,
            notes="Mẫu mơ hồ / Bỏ qua",
            action_type="skipped"
        )
        st.toast(f"Đã đánh dấu bỏ qua mẫu #{st.session_state.sample_index + 1}", icon="⏭️")
        if st.session_state.sample_index < len(filtered_samples) - 1:
            st.session_state.sample_index += 1
        st.rerun()

    if submit_quarantine:
        manager.save_annotation(
            sample_id=current_id,
            annotator_id=st.session_state.annotator_id,
            primary_label=selected_primary,
            sub_label=selected_sub,
            valence=selected_valence,
            arousal=selected_arousal,
            congruence=selected_congruence,
            crisis_level=3,
            notes=f"Cách ly khẩn cấp: {annotator_notes}",
            action_type="quarantined"
        )
        st.toast(f"Đã chuyển mẫu #{st.session_state.sample_index + 1} vào khu vực Cách Ly Khủng Hoảng!", icon="🚨")
        if st.session_state.sample_index < len(filtered_samples) - 1:
            st.session_state.sample_index += 1
        st.rerun()
