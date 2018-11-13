from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2, imp, io, os

image = cv2.imread("image1.jpg")
(h, w) = image.shape[:2]
image = image.reshape((image.shape[0] * image.shape[1], 3))

clt = MiniBatchKMeans(n_clusters=5)
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))

cv2.imshow("image", np.hstack([image, quant]))
cv2.imwrite("image.jpg", image)
cv2.waitKey(0)
