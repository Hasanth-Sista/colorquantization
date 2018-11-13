from skimage import io
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('image1.jpg')
io.imshow(image)
io.show()

rows = image.shape[0]
cols = image.shape[1]

image = image.reshape(image.shape[0] * image.shape[1], 3)
kmeans = KMeans(n_clusters=128, n_init=10, max_iter=200)
kmeans.fit(image)

clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
labels = np.asarray(kmeans.labels_, dtype=np.uint8)
labels = labels.reshape(rows, cols)

np.save('codebook_tiger.npy', clusters)
io.imsave('compressed_tiger.png', labels)

from skimage import io
import numpy as np

centers = np.load('codebook_tiger.npy')
c_image = io.imread('compressed_tiger.png')

image = np.zeros((c_image.shape[0], c_image.shape[1], 3), dtype=np.uint8)
for i in range(c_image.shape[0]):
    for j in range(c_image.shape[1]):
        image[i, j, :] = centers[c_image[i, j], :]
io.imsave('reconstructed_tiger.png', image)
io.imshow(image)
io.show()


# from sklearn.cluster import MiniBatchKMeans
# import numpy as np
# import argparse
# import cv2, imp, io, os
#
# image = cv2.imread("image1.jpg")
# (h, w) = image.shape[:2]
# image = image.reshape((image.shape[0] * image.shape[1], 3))
#
# clt = MiniBatchKMeans(n_clusters=5)
# labels = clt.fit_predict(image)
# quant = clt.cluster_centers_.astype("uint8")[labels]
# quant = quant.reshape((h, w, 3))
# image = image.reshape((h, w, 3))
#
# cv2.imshow("image", np.hstack([image, quant]))
# cv2.imwrite("image.jpg", image)
# cv2.waitKey(0)
