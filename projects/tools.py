import cv2


class coordinate:

    def mouseClicked(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"({x}, {y})")
