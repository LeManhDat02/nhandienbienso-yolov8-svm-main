import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Cấu hình matplotlib để hỗ trợ tiếng Việt
plt.rcParams['font.family'] = 'Arial'  # Nếu lỗi, bạn có thể đổi sang 'DejaVu Sans' hoặc 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def plot_metrics(results_folder):
    results_path = os.path.join(results_folder, 'results.csv')

    if not os.path.exists(results_path):
        print("❌ Không tìm thấy file results.csv!")
        return

    # Đọc dữ liệu từ results.csv
    import pandas as pd
    df = pd.read_csv(results_path)

    epochs = df.index + 1
    train_loss = df['train/loss']
    val_loss = df['val/loss']
    accuracy = df['metrics/accuracy_top1']

    # Vẽ Độ lỗi
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Độ lỗi huấn luyện (Train Loss)')
    plt.plot(epochs, val_loss, label='Độ lỗi kiểm tra (Validation Loss)')
    plt.xlabel('Số lần lặp (Epoch)')
    plt.ylabel('Độ lỗi')
    plt.title('Biểu đồ Độ lỗi theo Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # Vẽ Độ chính xác
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracy, label='Độ chính xác kiểm tra (Validation Accuracy)')
    plt.xlabel('Số lần lặp (Epoch)')
    plt.ylabel('Độ chính xác (%)')
    plt.title('Biểu đồ Độ chính xác theo Epochs')
    plt.legend()
    plt.grid()
    plt.show()

def plot_confusion_matrix(runs_folder):
    conf_matrix_path = os.path.join(runs_folder, 'confusion_matrix.png')
    if not os.path.exists(conf_matrix_path):
        print("❌ Không tìm thấy Confusion Matrix!")
        return

    img = plt.imread(conf_matrix_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.show()

if __name__ == "__main__":
    results_folder = 'runs/train2\confusion_matrix.png'
    plot_metrics(results_folder)
    plot_confusion_matrix(results_folder)