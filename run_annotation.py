"""
Emotion Journal AI - One-Click Annotation Studio Launcher
Khởi động nhanh ứng dụng gán nhãn đa phương thức Multimodal Emotion Studio.
Sử dụng: python run_annotation.py [--port 8501]
"""

import sys
import os
import subprocess
import argparse

# Thiết lập encoding UTF-8 chuẩn trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Khởi chạy Emotion Journal AI Annotation Studio")
    parser.add_argument("--port", type=int, default=8501, help="Cổng mạng máy chủ (Mặc định: 8501)")
    parser.add_argument("--headless", action="store_true", help="Chạy ở chế độ không tự mở trình duyệt")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_root, "scripts", "annotation_tool", "app.py")

    if not os.path.exists(app_path):
        print(f"[ERROR] Không tìm thấy file app: {app_path}")
        sys.exit(1)

    print("=" * 75)
    print("   EMOTION JOURNAL AI - MULTIMODAL ANNOTATION STUDIO")
    print("   Nền tảng gán nhãn Đa phương thức (Text + Voice 16kHz) chuẩn Khoa học")
    print("=" * 75)
    print(f"[*] Thư mục dự án: {project_root}")
    print(f"[*] Ứng dụng:       {app_path}")
    print(f"[*] Cổng mạng:      http://localhost:{args.port}")
    print("[*] Đang khởi động máy chủ Streamlit...")
    print("-" * 75)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        f"--server.port={args.port}",
    ]
    if args.headless:
        cmd.append("--server.headless=true")
    else:
        cmd.append("--server.headless=false")

    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n[*] Đã dừng máy chủ Annotation Studio an toàn.")

if __name__ == "__main__":
    main()
