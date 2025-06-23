# LicensePlateRecognition
License plate recognition using YOLOv8, Real-ERGAN, PaddleOCR

Pipeline:
1. Phát hiện biển số xe.
2. Cắt biển số xe.
3. Xác định 4 góc của biển số xe .
4. Sử dụng Perspective Transformation để vuông hóa biển số xe.
5. Sử dụng Real-ESRGAN để làm nét ảnh biển số xe giúp dễ dàng nhận diện ký tự hơn.
6. Sử dụng PaddleOCR để nhận diện các ký tự của biển số.