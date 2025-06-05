## 🧩 Mô tả bài toán

Dự án xây dựng một **ứng dụng nhận diện biển số xe tự động từ ảnh tĩnh** dựa trên mô hình học sâu kết hợp học máy truyền thống, tích hợp trên giao diện người dùng Tkinter.

---

### 🚗 Bối cảnh & Động lực

Việc nhận dạng biển số xe một cách tự động ngày càng quan trọng trong các ứng dụng:
- 📍 Quản lý xe ra/vào bãi đỗ
- 🚧 Giám sát giao thông đô thị
- 🏫 Kiểm soát phương tiện tại cổng trường, nhà máy, chung cư

Tuy nhiên, phương pháp thủ công thường chậm, dễ sai và không đáp ứng được nhu cầu thời gian thực. Vì vậy, cần có một hệ thống:
- 🤖 Tự động phát hiện biển số
- 🔤 Nhận dạng ký tự chính xác
- 🖥 Hiển thị thông tin trực quan, dễ dùng

---

### 🔁 Quy trình hoạt động

```mermaid
graph LR
A[Người dùng chọn ảnh xe] --> B[YOLOv8 phát hiện biển số]
B --> C[Cắt & xử lý vùng biển số]
C --> D[Tách ký tự]
D --> E[SVM nhận dạng ký tự]
E --> F[Hiển thị kết quả trên giao diện]
