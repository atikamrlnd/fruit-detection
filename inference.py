import cv2
from ultralytics import YOLO

# ======== KONFIGURASI ========
MODEL_PATH = "best.pt"
IMAGE_PATH = "buah.jpeg"
# ==============================

def main():
    model = YOLO(MODEL_PATH)
    results = model(IMAGE_PATH)

    annotated_frame = results[0].plot()

    cv2.imshow("Fruit Detection Result", annotated_frame)
    print("Tekan tombol apapun di window gambar untuk menutup...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
