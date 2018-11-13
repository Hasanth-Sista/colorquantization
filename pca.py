from sklearn.decomposition import PCA
from pylab import *
from skimage import data, io, color

link = "image1.jpg"
image = io.imread(link)

# subplot(2, 2, 1)
# io.imshow(image)
# xlabel('Original Image')

for i in range(1, 2):
    n_comp = 1 ** i
    pca = PCA(n_components=n_comp)
    image = image.reshape(image.shape[0] * image.shape[1], 3)
    pca.fit(image)
    coke_gray_pca = pca.fit_transform(image)

    coke_gray_restored = pca.inverse_transform(coke_gray_pca)
    # subplot(2, 2, i+1)
    io.imshow(coke_gray_restored)
    # xlabel('Restored image n_components = %s' %n_comp)
    # print('Variance retained %s %%' %((1 - sum(pca.explained_variance_ratio_) / size(pca.explained_variance_ratio_)) * 100))
    # print('Compression Ratio %s %%' %(float(size(coke_gray_pca)) / size(image) * 100))
    # io.imsave(coke_gray_restored, 'image.jpg')
    io.imsave('./colorquantization/', coke_gray_restored)