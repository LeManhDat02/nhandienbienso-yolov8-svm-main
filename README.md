🚗 Vehicle License Plate Recognition GUI - Nhóm 03
Ứng dụng giao diện đồ họa bằng Tkinter để nhận diện biển số xe từ hình ảnh sử dụng mô hình YOLOv8 để phát hiện biển số, sau đó sử dụng SVM để nhận diện ký tự trên biển số.
📌 Mô tả bài toán
Mục tiêu: Xây dựng hệ thống nhận diện biển số xe từ hình ảnh đầu vào (ảnh chụp xe) thông qua giao diện người dùng.

Hệ thống cần thực hiện các chức năng:

Cho phép người dùng chọn ảnh đầu vào.

Dùng YOLOv8 để phát hiện vị trí biển số xe trong ảnh.

Cắt và xử lý vùng ảnh chứa biển số.

Dùng SVM (Support Vector Machine) để nhận dạng từng ký tự trên biển số.

Hiển thị biển số nhận diện được lên giao diện.
🛠 Công nghệ sử dụng
Python 3.x

Tkinter (Giao diện GUI)

OpenCV (Xử lý ảnh)

PIL (Xử lý ảnh trong Tkinter)

Ultralytics YOLOv8 (Phát hiện biển số xe)

SVM (Nhận dạng ký tự, huấn luyện với tập dữ liệu ký tự biển số)

joblib (Lưu/trích xuất mô hình học máy)
🖼 Giao diện người dùng
Giao diện gồm:

Khung hiển thị ảnh gốc và ảnh biển số đã cắt.

Nút "Chọn ảnh" để tải ảnh từ máy tính.

Nút "Tìm biển số xe" để bắt đầu quá trình phát hiện và nhận diện.

Nhãn hiển thị ngày giờ và kết quả biển số xe.
⚙️ Cấu trúc hoạt động
Người dùng chọn ảnh bằng nút "Chọn ảnh".

Ảnh được hiển thị và ghi lại thời gian thêm.

Nhấn nút "Tìm biển số xe" để:

YOLOv8 phát hiện biển số trong ảnh.

Cắt vùng biển số và hiển thị.

Dùng SVM nhận dạng từng ký tự.

Hiển thị biển số nhận diện được lên giao diện.
![image](https://github.com/user-attachments/assets/9117f46b-8d04-4733-95bb-dc581d629f8f)
![image](https://github.com/user-attachments/assets/e21f67f8-cc40-423a-99bb-257cf31e41fe)

