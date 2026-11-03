"""Hand label handling.

The frame is mirrored (cv2.flip) before MediaPipe sees it, and MediaPipe expects a mirrored
(selfie) image, so its labels already name the person's own hands. An earlier swap in the
pipeline made every label the opposite of the real hand. Set HANDEDNESS_SWAP=on to bring the
old swap back (for comparing with sessions recorded before 2 Nov 2026).
"""
import os


def swap_enabled():
    return os.environ.get("HANDEDNESS_SWAP", "off").lower() == "on"


def final_label(raw_label):
    if swap_enabled():
        return "Right" if raw_label == "Left" else "Left"
    return raw_label


def raw_label_for(final):
    # swapping is its own inverse
    return final_label(final)


def describe():
    return {"swap": swap_enabled()}
