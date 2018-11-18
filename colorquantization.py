from skimage import io
from sklearn.cluster import KMeans
import numpy as np


for i in range(1, 4):
    image = io.imread('image'+str(i)+'.jpg')
    io.imshow(image)
    io.show()

    rows = image.shape[0]
    cols = image.shape[1]

    image = image.reshape(image.shape[0] * image.shape[1], 3)
    kmeans = KMeans(n_clusters=10, n_init=10, max_iter=200)
    kmeans.fit(image)

    clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
    labels = np.asarray(kmeans.labels_, dtype=np.uint8)
    labels = labels.reshape(rows, cols)

    np.save('npyfiles/quantized_image_code'+str(i)+'.npy', clusters)
    io.imsave('npyfiles/quantized_image'+str(i)+'.png', labels)

    centers = np.load('npyfiles/quantized_image_code'+str(i)+'.npy')
    c_image = io.imread('npyfiles/quantized_image'+str(i)+'.png')

    image = np.zeros((c_image.shape[0], c_image.shape[1], 3), dtype=np.uint8)

    for j in range(c_image.shape[0]):
        for k in range(c_image.shape[1]):
            image[j, k, :] = centers[c_image[j, k], :]

    io.imsave('quantizedImages/quantized'+str(i)+'.png', image)
    io.imshow(image)
    io.show()
