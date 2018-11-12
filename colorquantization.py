from skimage import io
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('image1.jpg')
rows = image.shape[0]
cols = image.shape[1]
image = image.reshape(image.shape[0] * image.shape[1], 3)
kmeans = KMeans(n_clusters=5, max_iter=300)
kmeans.fit(image)

clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)

labels = np.asarray(kmeans.labels_, dtype=np.uint8)
labels = labels.reshape(rows, cols)


np.save('codebook_tiger.npy', clusters)
io.imsave('compressed_tiger.jpg', labels)


centers = np.load('codebook_tiger.npy')

c_image = io.imread('compressed_tiger.jpg')
print(c_image.shape[0])
# image = np.zeros((c_image.shape[0], c_image.shape[1], 3))
#
# for i in range(c_image.shape[0]):
#     for j in range(c_image.shape[1]):
#         image[i, j, :] = centers[c_image[i, j], :]
#
# io.imsave('reconstructed_tiger.jpg', image)

