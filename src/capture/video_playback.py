import cv2


def play_video(video_path):

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frame count: {frame_count}")

    while True:

        success, frame = video.read()

        if not success:
            break

        cv2.putText(
                    frame,
                    "REPLAY...",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

        cv2.imshow("Saved Video", frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

    print("Replaying saved video...")

    video.release()
    cv2.destroyAllWindows()

    print("Playback finished.")