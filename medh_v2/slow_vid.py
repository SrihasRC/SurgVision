import os
import subprocess

def slowdown_video():
    filename = input("Enter MP4 filename (with .mp4): ").strip()
    rate = float(input("Enter slowdown rate (e.g., 2 for 2x slower): ").strip())

    if not os.path.isfile(filename):
        print("File not found!")
        return

    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_slowed_{rate}x{ext}"

    # Video slowdown: setpts multiplies timestamps
    # Audio slowdown: atempo supports 0.5–2.0 range
    video_filter = f"setpts={rate}*PTS"

    if 0.5 <= 1/rate <= 2.0:
        audio_filter = f"atempo={1/rate}"
    else:
        print("Slowdown rate too extreme for audio processing.")
        return

    command = [
        "ffmpeg",
        "-i", filename,
        "-filter:v", video_filter,
        "-filter:a", audio_filter,
        output_filename
    ]

    subprocess.run(command)

    print(f"\nSaved slowed video as: {output_filename}")

if __name__ == "__main__":
    slowdown_video()
