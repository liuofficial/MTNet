import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import scipy.signal as sg
import os
import shutil
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def generateRandomList(numlist: list, maxNum, count):
    '''
    produce needed random list
    :param numlist: random list
    :param maxNum: the max number
    :param count: the count
    :return:
    '''
    i = 0
    while i < count:
        num = random.randint(1, maxNum)
        if num not in numlist:
            numlist.append(num)
            i += 1

def standard(X):
    '''
    Standardization
    universal
    :param X:
    :return:
    '''
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return np.float32(X)

def Fill_B(B, h, w):
    '''
    change the size of Blur matrix B to meet the blur operation
    since the size of estimate B is 10*10
    :param B:
    :param h:
    :param w:
    :return:
    '''
    tB = np.zeros([h, w], dtype=np.float32)
    tB[-4:, -4:] = B[-4:, -4:]
    tB[:6, -4:] = B[:6, -4:]
    tB[-4:, :6] = B[-4:, :6]
    tB[:6, :6] = B[:6, :6]
    return tB

def sumtoOne(R):
    '''
    R needs to be normalized, which is divided by the sum of the rows (the number of bands in hrhs)
    :param R:
    :return:
    '''
    div = np.sum(R, axis=1)
    div = np.expand_dims(div, axis=-1)
    R = R / div
    return R

def checkFile(path):
    '''
    if filepath not exist make it
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def getSpectralResponse():
    '''
    spectral response function for CAVE and HARVARD
    :return:
    '''
    R = np.array(
        [[2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    div = np.sum(R, axis=1)
    div = np.expand_dims(div, axis=-1)
    R = R / div
    return R


def getBlurMatrix(kernal_size, sigma):
    '''
    get Blur matrix B
    :param kernal_size:
    :param sigma:
    :return:
    '''
    side = cv2.getGaussianKernel(kernal_size, sigma)
    Blur = np.multiply(side, side.T)
    return Blur


def get_kernal(kernal_size, sigma, rows, cols):
    '''
    Generate a Gaussian kernel and make a fast Fourier transform
    :param kernal_size:
    :param sigma:
    :return:
    '''
    # Generate 2D Gaussian filter
    blur = cv2.getGaussianKernel(kernal_size, sigma) * cv2.getGaussianKernel(kernal_size, sigma).T
    psf = np.zeros([rows, cols])
    psf[:kernal_size, :kernal_size] = blur
    # Cyclic shift, so that the Gaussian core is located at the four corners
    B1 = np.roll(np.roll(psf, -kernal_size // 2, axis=0), -kernal_size // 2, axis=1)
    # Fast Fourier Transform
    fft_b = np.fft.fft2(B1)
    # return fft_b
    return fft_b


def spectralDegrade(X, R, addNoise=True, SNR=40):
    '''
    spectral downsample
    :param X:
    :param R:
    :return:
    '''
    height, width, bands = X.shape
    X = np.reshape(X, [-1, bands], order='F')
    Z = np.dot(X, R.T)
    Z = np.reshape(Z, [height, width, -1], order='F')

    if addNoise:
        h, w, c = Z.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Z)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Z += sigmah * np.random.randn(h, w, c)

    return Z


def Blurs(X, B, addNoise=True, SNR=30):
    '''
    downsample using fft
    :param X:
    :param B:
    :return:
    '''
    B = np.expand_dims(B, axis=-1)
    Y = np.real(np.fft.ifftn(np.fft.fftn(X) * B))

    if addNoise:
        h, w, c = Y.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Y)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Y += sigmah * np.random.randn(h, w, c)
    return Y


def upSample(X, ratio=8):
    '''
    upsample using cubic
    :param X:
    :param ratio:
    :return:
    '''
    h, w, c = X.shape
    return cv2.resize(X, (w * ratio, h * ratio), interpolation=cv2.INTER_CUBIC)


def downSample(X, B, ratio, addNoise=True, SNR=30):
    '''
    downsample using fft
    :param X:
    :param B:
    :param ratio:
    :return:
    '''

    B = np.expand_dims(B, axis=-1)
    Y = np.real(np.fft.ifftn(np.fft.fftn(X) * B))

    if addNoise:
        h, w, c = Y.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Y)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Y += sigmah * np.random.randn(h, w, c)

    # downsample
    Y = Y[::ratio, ::ratio, :]
    return Y


def downSample1(X, B, ratio, rows,cols,kernal_size):
    '''
    downsample using fft
    :param X:
    :param B:
    :param ratio:
    :return:
    '''

    psf = np.zeros([rows, cols])
    psf[:kernal_size, :kernal_size] = B
    # Cyclic shift, so that the Gaussian core is located at the four corners
    B1 = np.roll(np.roll(psf, -kernal_size // 2, axis=0), -kernal_size // 2, axis=1)
    # Fast Fourier Transform
    fft_b = np.fft.fft2(B1)

    B2 = np.expand_dims(fft_b, axis=-1)
    Y = np.real(np.fft.ifftn(np.fft.fftn(X) * B2))

    # downsample
    Y = Y[::ratio, ::ratio, :]
    return Y


def readCAVEData(path, mat_path):
    '''
    Read initial CAVE data
    since the original data is standardized we do not repeat it
    :return:
    '''
    hsi = np.zeros([512, 512, 31], dtype=np.float32)
    checkFile(mat_path)
    count = 0
    for dir in os.listdir(path):
        concrete_path = path + '/' + dir + '/' + dir
        for i in range(31):
            fix = str(i + 1)
            if i + 1 < 10:
                fix = '0' + str(i + 1)
            png_path = concrete_path + '/' + dir + '_' + fix + '.png'
            try:
                hsi[:, :, i] = plt.imread(png_path)
            except:
                img = plt.imread(png_path)
                img = img[:, :, :3]
                img = np.mean(img, axis=2)
                hsi[:, :, i] = img

        count += 1
        print('%d has finished' % count)
        sio.savemat(mat_path + str(count) + '.mat', {'HS': hsi})

def readPCData(path,mat_path):
    checkFile(mat_path)
    pavia = sio.loadmat(path)['pavia']
    pavia = standard(pavia)
    r, c, _ = pavia.shape # 610,340
    pavia_1 = pavia[:960, :640, :]
    sio.savemat(mat_path + 'pc.mat', {'HS': pavia_1})
    print('done')

def readBotswanaData(path, mat_path):
    checkFile(mat_path)
    data = sio.loadmat(path)['Botswana']
    data = standard(data)
    r, c, _ = data.shape # 610,340
    data_1 = data[:1200, :240, :]
    sio.savemat(mat_path + '1.mat', {'HS': data_1})
    print('done')

def createSimulateData(data_index, B, ratio):
    '''
    create simulated data
    :param data_index:
    :param B:
    :param ratio
    :return:
    '''
    if data_index == 0:
        # CAVE
        mat_path = r'E:/Datasets/CAVE/CAVEMAT/'
        num_start = 1
        num_end = 32

    elif data_index == 1:
        # PC
        mat_path = r'E:/Datasets/PC/PCMAT/'
        num_start = 1
        num_end = 24

    elif data_index == 2:
        # Bostwana
        mat_path = r'E:/Datasets/Botswana/BTSMAT/'
        num_start = 1
        num_end = 20

    elif data_index == 3:
        # UH
        mat_path = r'E:/Datasets/Houstan_mat/'
        num_start = 1
        num_end = 3
        save_path = 'E:/DATA/houstanmat/'

    if data_index == 0:
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            hs = mat['HS']
            pan = np.mean(hs,axis=2)
            pan = np.expand_dims(pan,axis=2)
            lrhs = downSample(hs, B, ratio, False)  # generate the LR-HSI
            sio.savemat(mat_path + str(i) + '.mat', {'label': hs, 'P': pan, 'Y': lrhs})
            print('%d has finished' % i)

    elif data_index in [1,2]:
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            hs = mat['HS']
            pan = np.mean(hs,axis=2)
            pan = np.expand_dims(pan,axis=2)
            lrhs = downSample(hs, B, ratio,False)
            sio.savemat(mat_path + str(i) + '.mat', {'label': hs, 'Y': lrhs,'P':pan})
            print('%d has finished' % i)

    elif data_index == 3:
        BRpath = 'E:/UH/Houston/'
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            print(mat.keys())
            hs = mat['Lrhs']
            ms1 = mat['ms']
            BRdata = sio.loadmat(BRpath + 'BR' +str(i) + '.mat')
            B = BRdata['B']
            lrhs = downSample1(hs, B, ratio,rows=128,cols=128,kernal_size=ratio)
            ms = cv2.resize(ms1,dsize=(128,128))
            pan = np.mean(ms,axis=2)
            pan = np.expand_dims(pan,axis=2)
            sio.savemat(save_path + str(i) + '.mat', {'label': hs, 'Y': lrhs,'P':pan})
            print('%d has finished' % i)

def cutTrainingPiecesForSimulatedDataset(data_index):
    '''
    produce training pieces
    :param train_index:
    :return:
    '''
    if data_index == 0:
        # CAVE
        # the first 20 images are patched for training and verifying
        piece_size = 48
        stride = 16
        rows, cols = 512, 512
        num_start = 1
        num_end = 20
        mat_path = r'E:/DATA/CAVEMAT/'
        count = 0
        ratio = 8
        piece_save_path = 'E:/Datasets/CAVE/CAVE_48_patch_r88/train/'

    elif data_index == 1:
        # PC
        # the first 16 images are patched for training and verifying
        piece_size = 48
        stride = 4
        rows, cols = 160, 160
        num_start = 1
        num_end = 16
        mat_path = r'E:/Datasets/PC/PCMAT/'
        count = 0
        ratio = 8
        piece_save_path = 'E:/Datasets/PC/PC_48_patch/train/'

    elif data_index == 2:
        piece_size = 48
        stride = 2
        rows, cols = 120, 120
        num_start = 1
        num_end = 14
        mat_path = r'E:/Datasets/Botswana/BTSMAT/'
        count = 0
        ratio = 8
        piece_save_path = 'E:/Datasets/Botswana/BS_48_patch/train/'

    elif data_index == 3:
        piece_size = 32
        stride = 1
        rows, cols = 128, 128
        num_start = 1
        num_end = 2
        mat_path = r'E:/DATA/houstanmat/'
        count = 0
        ratio = 8
        piece_save_path = 'E:/Datasets/UH/UH_32_patch/train/'

    checkFile(piece_save_path)
    if data_index in [0, 1]:
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            print(mat.keys())
            X = mat['label']
            Y = mat['Y']
            P = mat['P']
            for x in range(0, rows - piece_size + stride, stride):
                for y in range(0, cols - piece_size + stride, stride):
                    Y_piece = Y[x // ratio:(x + piece_size) // ratio, y // ratio:(y + piece_size) // ratio, :]
                    P_piece = P[x:x + piece_size, y:y + piece_size, :]
                    label_piece = X[x:x + piece_size, y:y + piece_size, :]

                    sio.savemat(piece_save_path + 'a%d.mat' % count,
                                {'Y': Y_piece,  'X': label_piece,'P':P_piece})
                    count += 1
                    print('piece num %d has saved' % count)
            print('%d has finished' % i)

    elif data_index == 2:
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            X = mat['label']
            Y = mat['Y']
            P = mat['P']
            for x in range(0, rows - piece_size + stride, stride):
                for y in range(0, cols - piece_size + stride, stride):
                    Y_piece = Y[x // ratio:(x + piece_size) // ratio, y // ratio:(y + piece_size) // ratio, :]
                    label_piece = X[x:x + piece_size, y:y + piece_size, :]
                    p_piece = P[x:x + piece_size, y:y + piece_size, :]
                    sio.savemat(piece_save_path + 'a%d.mat' % count,
                                {'Y': Y_piece, 'X': label_piece,'P':p_piece})
                    count += 1
                    print('piece num %d has saved' % count)
            print('%d has finished' % i)

    elif data_index == 3:
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            X = mat['label']
            Y = mat['Y']
            P = mat['P']
            for x in range(0, rows - piece_size + stride, stride):
                for y in range(0, cols - piece_size + stride, stride):
                    Y_piece = Y[x // ratio:(x + piece_size) // ratio, y // ratio:(y + piece_size) // ratio, :]
                    label_piece = X[x:x + piece_size, y:y + piece_size, :]
                    p_piece = P[x:x + piece_size, y:y + piece_size, :]
                    sio.savemat(piece_save_path + 'a%d.mat' % count,
                                {'Y': Y_piece, 'X': label_piece,'P':p_piece})
                    count += 1
                    print('piece num %d has saved' % count)
            print('%d has finished' % i)
    return count


def generateVerticationSet(mat_path, num):
    '''
    Randomly select 20% as the verification set
    :param train_path:
    :param verti_path:
    :param num:
    :return:
    '''
    ratio = 0.2
    verti_num = int(num * ratio)
    num_list = []
    train_path = mat_path + '/train/'
    verti_path = mat_path + '/valid/'
    checkFile(verti_path)
    generateRandomList(num_list, num, verti_num)
    print(num_list.__len__())
    for ind, val in enumerate(num_list):

        try:
            shutil.copy(train_path + 'a%d.mat' % val, verti_path + '%d.mat' % (ind + 1))
            os.remove(train_path + 'a%d.mat' % val)
            print('%d has created' % (ind + 1))
        except:
            print('raise error')
    print('veticatication set created')
    print('done rerank train set')
    # rename the left train pieces
    reRankfile(train_path, '')
    return num_list.__len__(), num - num_list.__len__()


def reRankfile(path, name):
    '''
    Reorder mat by renaming
    :param path:
    :param name:
    :return:
    '''
    count = 0
    file_list = os.listdir(path)
    for file in file_list:
        try:
            count += 1
            newname = str(count)
            print(newname)
            os.rename(path + file, path + '%s.mat' % (newname))
        except:
            print('error')


if __name__ == '__main__':
    # manipulating datasets including CAVE, PC, Bostwana, University of Houston
    data_index = 0 # 0, 1 ,2, 3 represents the four datasets, respectively

    if data_index == 0:
        # CAVE
        path = 'E:/DATA/CAVE/'  # replace with your path which puts the downloading CAVE data
        mat_path = r'E:/Datasets/CAVE/CAVEMAT/'  # the path of saving the entile HSI with .mat format
        # convert the data into .mat format
        readCAVEData(path, mat_path)
        #
        # # produce the simulated data according to the Wald's protocol
        B = get_kernal(8, 2, 512, 512)  # the blurring kernel with size of 8*8 and a standard deviation of 2
        # # The spectral response matrix coming from Nikon Camera (400-700 nm)
        ratio = 8  # the spatial resolution ratio
        createSimulateData(data_index, B, ratio=ratio)

        # produce the training and verification pieces
        count = cutTrainingPiecesForSimulatedDataset(data_index)
        # randomly select 20% of the training pieces as the verification pieces
        generateVerticationSet('E:/Datasets/CAVE/CAVE_48_patch/', count)

    elif data_index == 1:
        # PC
        path = r'E:/Datasets/Pavia.mat' # replace with your path which puts the downloading Harvard data
        mat_path = r'E:/Datasets/PC/'  # the path of saving the entile HSI with .mat format
        # # convert the data into .mat format
        readPCData(path,mat_path)

        ## produce the simulated data according to the Wald's protocol
        B = get_kernal(8, 2, 160, 160)  # the blurring kernel with size of 8*8 and a standard deviation of 2
        # # The spectral response matrix coming from Nikon Camera (400-700 nm)
        ratio = 8  # the spatial resolution ratio
        createSimulateData(data_index, B, ratio=ratio)

        #  produce the training and verification pieces
        count = cutTrainingPiecesForSimulatedDataset(data_index)
        #  randomly select 20% of the training pieces as the verification pieces
        generateVerticationSet('E:/Datasets/PC/PC_48_patch/', count)


    elif data_index == 2:
        # bostwana
        path = r'E:/Datasets/Botswana.mat'  # replace with your path which puts the downloading Harvard data
        mat_path = r'E:/Datasets/Botswana/'  # the path of saving the entile HSI with .mat format
        # # convert the data into .mat format
        readBotswanaData(path,mat_path)
        B = get_kernal(8, 2, 120, 120)  # the blurring kernel with size of 8*8 and a standard deviation of 2
        ratio = 8  # the spatial resolution ratio
        createSimulateData(data_index, B, ratio=ratio)
        # produce the training and verification pieces
        count = cutTrainingPiecesForSimulatedDataset(data_index)
        # ## randomly select 20% of the training pieces as the verification pieces
        generateVerticationSet('E:/Datasets/Botswana/BS_48_patch/', count)

    elif data_index == 3:
        # UH
        # produce the training and verification pieces
        B = get_kernal(8, 2, 128, 128)  # This B will not be involved in the production of the UH dataset
        ratio = 8
        createSimulateData(data_index, B, ratio=ratio)
        count = cutTrainingPiecesForSimulatedDataset(data_index)
        # ## randomly select 20% of the training pieces as the verification pieces
        generateVerticationSet('E:/Datasets/UH/UH_32_patch/', count)
