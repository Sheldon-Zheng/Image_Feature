import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from skimage import feature as skft

class Texture():
  def __init__(self):
    self.radius = 1
    self.n_point = self.radius * 8

  def rgb2gray(self, A):
    # 定义一个空列表
    x = []

    for i in range(A.shape[1]):
      for j in range(A.shape[0]):
        # 使用转换公式进行彩色灰度转换
        x.append(np.int(A[i, j, 0] * 0.3 + A[i, j, 1] * 0.59 + A[i, j, 2] * 0.11))
    return x

  def processPicture(self, index):

    img = mpimg.imread('dataset/'+str(index)+'.tiff')
    ff = self.rgb2gray(A=img)
    # 将列表转化为矩阵
    m_mat = np.array(ff)
    # 将矩阵转变为原尺寸大小
    grayimg = m_mat.reshape(img.shape[0], img.shape[1])
    # 显示灰度图片
    plt.figure()
    plt.imshow(grayimg, cmap=plt.get_cmap('gray'))
    plt.savefig('dataset/'+str(index)+'.tiff')
    plt.show()

  def loadPicture(self):
    train_index = 0
    test_index = 0
    train_data = np.zeros((200, 171, 171))
    test_data = np.zeros((160, 171, 171))
    train_label = np.zeros(200)
    test_label = np.zeros(160)
    for i in np.arange(1, 4):
      image = mpimg.imread('dataset/'+str(i)+'.tiff')
      data = np.zeros((513, 513))
      data[0:image.shape[0], 0:image.shape[1]] = image
      index = 0
      for row in np.arange(3):
        for col in np.arange(3):
          if index < 5:
            train_data[train_index, :, :] = data[171*row:171*(row+1),171*col:171*(col+1)]
            train_label[train_index] = i
            train_index += 1
          else:
            test_data[test_index, :, :] = data[171*row:171*(row+1),171*col:171*(col+1)]
            test_label[test_index] = i
            test_index += 1
          index += 1

    return train_data, test_data, train_label, test_label

  def texture_detect(self):
    train_data, test_data, train_label, test_label = self.loadPicture()
    n_point = self.n_point
    radius = self.radius
    train_hist = np.zeros((200, 256))
    test_hist = np.zeros((160, 256))
    for i in np.arange(200):
      lbp=skft.local_binary_pattern(train_data[i], n_point, radius, 'default')
      max_bins = int(lbp.max() + 1)
      # hist size:256
      train_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))

    for i in np.arange(160):
      lbp = skft.local_binary_pattern(test_data[i], n_point, radius, 'default')
      max_bins = int(lbp.max() + 1)
      # hist size:256
      test_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))

    return train_hist, test_hist

  def classifer(self):
    train_data, test_data, train_label, test_label = self.loadPicture()
    train_hist, test_hist = self.texture_detect()
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    result = OneVsRestClassifier(svr_rbf, -1).fit(train_hist, train_label).score(test_hist, test_label)
    return result*100

if __name__ == '__main__':
  test = Texture()
  # 生成灰度图像处理部分
  # for i in range(1, 4):
  #   test.processPicture(index=i)

  # 计算分类准确度部分
  accuracy = test.classifer()
  print('Final Accuracy = '+str(accuracy)+'%')