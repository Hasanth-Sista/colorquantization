from sklearn.decomposition import PCA
import cv2

# give absolute path for image input
input = cv2.imread("image1.jpg")

output = "compressedImages/"

for i in (5, 25, 50):
    pca = PCA(n_components=i)
    shape = input.shape
    reshaped_image = input.reshape(shape[0], shape[1]*shape[2])

    reduced_image = pca.fit_transform(reshaped_image)
    reconstructed_image = pca.inverse_transform(reduced_image)

    cv2.imwrite(output+"image"+str(i)+".jpg", reconstructed_image.reshape(shape))
