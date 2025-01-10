import numpy as np
import cv2
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def process_image(image, vq_block_size, n_clusters):
    dct_block_size = 8  # 固定 DCT 區塊大小
    height, width = image.shape
    dct_blocks = []

    # DCT 轉換
    for i in range(0, height, dct_block_size):
        for j in range(0, width, dct_block_size):
            block = image[i:i+dct_block_size, j:j+dct_block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_blocks.append(dct_block)

    # 向量量化
    vq_dct_blocks = []
    for dct_block in dct_blocks:
        for i in range(0, dct_block_size, vq_block_size):
            for j in range(0, dct_block_size, vq_block_size):
                vq_block = dct_block[i:i+vq_block_size, j:j+vq_block_size]
                vq_dct_blocks.append(vq_block.flatten())

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vq_dct_blocks)
    quantized_blocks = kmeans.cluster_centers_[kmeans.labels_]

    # IDCT 還原
    idct_blocks = []
    idx = 0
    for dct_block in dct_blocks:
        idct_block = np.zeros_like(dct_block)
        for i in range(0, dct_block_size, vq_block_size):
            for j in range(0, dct_block_size, vq_block_size):
                block = quantized_blocks[idx].reshape((vq_block_size, vq_block_size))
                idct_block[i:i+vq_block_size, j:j+vq_block_size] = block
                idx += 1
        idct_block = idct(idct(idct_block.T, norm='ortho').T, norm='ortho')
        idct_blocks.append(idct_block)

    # 重建圖片
    idct_image = np.zeros_like(image)
    idx = 0
    for i in range(0, height, dct_block_size):
        for j in range(0, width, dct_block_size):
            idct_image[i:i+dct_block_size, j:j+dct_block_size] = idct_blocks[idx]
            idx += 1

    # 計算 PSNR 和 BPP
    psnr = peak_signal_noise_ratio(image, idct_image)
    bpp = np.log2(n_clusters) / (vq_block_size ** 2)

    return idct_image, psnr, bpp

# 讀取圖片
image_path = "./indigenous.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))

# 設定不同的參數組合
params = [
    (8, 128),
    (8, 1024),
    (4, 32),
    (4, 256),
    (4, 1024)

]

# 處理每個參數組合
results = []
for vq_block_size, n_clusters in params:
    idct_image, psnr, bpp = process_image(image, vq_block_size, n_clusters)
    results.append((idct_image, psnr, bpp))

# 顯示結果
fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
for ax, (idct_image, psnr, bpp), (vq_block_size, n_clusters) in zip(axes, results, params):
    ax.imshow(idct_image, cmap='gray')
    ax.set_title(f"VQ: {vq_block_size}x{vq_block_size}\nClusters: {n_clusters}\nPSNR: {psnr:.4f} dB\nBPP: {bpp:.4f}")
    ax.axis('off')

plt.tight_layout()

# 儲存整個顯示的圖像
plt.savefig("combined_output.png")

plt.show()

# 新增：創建和顯示表格
table_data = []
for (vq_block_size, n_clusters), (_, psnr, bpp) in zip(params, results):
    table_data.append([n_clusters, vq_block_size, bpp, psnr])

df = pd.DataFrame(table_data, columns=['k', 'b', 'BPP', 'PSNR'])
print(df.to_string(index=False))
