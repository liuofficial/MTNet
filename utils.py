from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import scipy.io as sio
import torch
import os
from Data_processing import checkFile

class Dataset(torch.utils.data.Dataset):

    def __init__(self, mat_path):
        self.data_path = mat_path
        self.images = self.read_file()

    def __getitem__(self, index):
        data = sio.loadmat(self.images[index])
        pan = data['P']
        lrhs = data['Y']
        hs = data['X']
        lrhs = np.transpose(lrhs, (2, 0, 1)).astype(np.float32)
        lrhs = torch.from_numpy(lrhs)
        pan = np.transpose(pan, (2, 0, 1)).astype(np.float32)
        pan = torch.from_numpy(pan)
        hs = np.transpose(hs, (2, 0, 1)).astype(np.float32)
        hs = torch.from_numpy(hs)
        return {'Z': pan, 'Y': lrhs, 'X': hs}

    def __len__(self):
        return len(self.images)

    def read_file(self):
        path_list = []
        for ph in os.listdir(self.data_path):
            path = self.data_path + ph
            path_list.append(path)
        return path_list

def PSNR_GPU(labels, output):
    s = labels.shape[0]
    labels = labels.cpu().squeeze().detach().numpy()
    output = output.cpu().squeeze().detach().numpy()
    if (s == 1):
        labels = np.transpose(labels, (1, 2, 0))
        output = np.transpose(output, (1, 2, 0))
    else:
        labels = np.transpose(labels, (0, 2, 3, 1))
        output = np.transpose(output, (0, 2, 3, 1))

    return peak_signal_noise_ratio(labels, output)

def cutCAVEPieces_Test(mpath,msave_path):
    piece_size = 128
    stride = piece_size
    rows, cols = 512, 512
    num_start = 21
    num_end = 32
    ratio = 8
    mat_path = mpath
    count = 0
    piece_save_test = msave_path
    checkFile(piece_save_test)
    for i in range(num_start, num_end + 1):
        mat = sio.loadmat(mat_path + '%d.mat' % i)
        X = mat['label']
        Z = mat['P']
        Y = mat['Y']
        for x in range(0, rows - piece_size + stride, stride):
            for y in range(0, cols - piece_size + stride, stride):
                data2_piece = Z[x:x + piece_size, y:y + piece_size, :]
                label_piece = X[x:x + piece_size, y:y + piece_size, :]
                data1_piece = Y[x // ratio:(x + piece_size) // ratio, y // ratio:(y + piece_size) // ratio, :]
                count += 1
                sio.savemat(piece_save_test + '%d.mat' % count,
                            {'Y': data1_piece, 'Z': data2_piece, 'X': label_piece})
                print('piece num %d has saved' % count)
        print('%d has finished' % i)
    # print(count)
    print('done')
    return count

def merge_Cave_test(tpath,spath):
    piece_size = 128
    stride = piece_size
    rows, cols = 512, 512
    mat_path = tpath
    count = 0
    save_path = spath
    checkFile(save_path)
    for i in range(0,12):
        HS = np.zeros((512,512,31))
        j = 16 * i + 1
        for x in range(0, rows - piece_size + stride, stride):
            for y in range(0, cols - piece_size + stride, stride):
                mat = sio.loadmat(mat_path + '%d.mat' % j)
                z = mat['hs']
                HS[x:x + piece_size, y:y + piece_size, :] = z
                j = j + 1
        sio.savemat(save_path + str(21+i) + '.mat',{'hs': HS})
    print(count)
    print('done')


def quality_accessment(out:dict,reference, target, ratio):
    '''
    融合质量评价
    :param references:参照图像
    :param target: 融合图像
    :param ratio:边界大小
    :return:
    '''
    rows, cols, bands = reference.shape
    # 去除边界
    # removed_reference = reference[ratio:rows - ratio, ratio:cols - ratio, :]
    # removed_target = target[ratio:rows - ratio, ratio:cols - ratio, :]
    out['cc'] = CC(reference, target)
    out['sam'] = SAM(reference, target)[0]
    out['rmse'] = RMSE(reference, target)
    out['egras'] = ERGAS(reference, target, ratio)
    out['psnr'] = PSNR(reference, target)
    out['ssim'] = SSIM(reference, target)
    return out



def CC(reference, target):
    '''
    相关性评价(按通道求两者相关系数，再取均值，理想值为1)
    :param references: 参照图像
    :param target: 融合图像
    :return:
    '''
    bands = reference.shape[2]
    out = np.zeros([bands])
    for i in range(bands):
        ref_temp = reference[:, :, i].flatten(order='F')  # 展开成向量
        target_temp = target[:, :, i].flatten(order='F')  # 展开成向量
        cc = np.corrcoef(ref_temp, target_temp)  # 求取相关系数矩阵
        out[i] = cc[0, 1]
    return np.mean(out)


def dot(m1, m2):
    '''
    两个三维图像求相同位置不同通道构成的向量内积
    :param m1: 图像1
    :param m2: 图像2
    :return:
    '''
    r, c, b = m1.shape
    p = r * c
    temp_m1 = np.reshape(m1, [p, b], order='F')
    temp_m2 = np.reshape(m2, [p, b], order='F')
    out = np.zeros([p])
    for i in range(p):
        out[i] = np.inner(temp_m1[i, :], temp_m2[i, :])
    out = np.reshape(out, [r, c], order='F')
    return out


def SAM(reference, target):
    '''
    光谱角度映射器评价（求取平均光谱映射角度，理想值为0）
    :param reference: 参照图像
    :param target: 融合图像
    :return:
    '''
    rows, cols, bands = reference.shape
    pixels = rows * cols
    eps = 1 / (2 ** 52)  # 浮点精度
    prod_scal = dot(reference, target)  # 取各通道相同位置组成的向量进行内积运算
    norm_ref = dot(reference, reference)
    norm_tar = dot(target, target)
    prod_norm = np.sqrt(norm_ref * norm_tar)  # 二范数乘积矩阵
    prod_map = prod_norm
    prod_map[prod_map == 0] = eps  # 除法避免除数为0
    # print(prod_scal/prod_map)
    map = np.arccos(prod_scal / prod_map)  # 求得映射矩阵
    # print(map)
    prod_scal = np.reshape(prod_scal, [pixels, 1])
    prod_norm = np.reshape(prod_norm, [pixels, 1])
    z = np.argwhere(prod_norm == 0)[:0]  # 求得prod_norm中为0位置的行号向量
    # 去除这些行，方便后续进行点除运算
    prod_scal = np.delete(prod_scal, z, axis=0)
    prod_norm = np.delete(prod_norm, z, axis=0)
    # 求取平均光谱角度
    angolo = np.sum(np.arccos(prod_scal / prod_norm)) / prod_scal.shape[0]
    # 转换为度数
    angle_sam = np.real(angolo) * 180 / np.pi
    return angle_sam, map

def SSIM_BAND(reference, target):
    return compare_ssim(reference,target,data_range=1.0)


def SSIM(reference, target):
    '''
    平均结构相似性
    :param reference:
    :param target:
    :return:
    '''
    rows,cols,bands = reference.shape
    mssim = 0
    for i in range(bands):
        mssim += SSIM_BAND(reference[:,:,i],target[:,:,i])
    mssim /= bands
    return mssim
    # return compare_ssim(reference, target, multichannel=True)


def PSNR(reference, target):
    '''
    峰值信噪比
    :param reference: 参照图像
    :param target: 融合图像
    :return:
    '''
    max_pixel = 1.0
    return 10.0 * np.log10((max_pixel ** 2) / np.mean(np.square(reference - target)))
    # return compare_psnr(reference, target)


def RMSE(reference, target):
    '''
    根均方误差评价（两图像各位置像素值差的F范数除以总像素个数的平方根，理想值为0）
    :param reference: 参照图像
    :param target: 融合图像
    :return:
    '''
    rows, cols, bands = reference.shape
    pixels = rows * cols * bands
    out = np.sqrt(np.sum((reference - target) ** 2) / pixels)
    return out


def ERGAS(references, target, ratio):
    '''
    总体相对误差评价（各通道求取相对均方误差取根均值，再乘以相应系数，理想值为0）
    :param references: 参照图像
    :param target: 融合图像
    :return:
    '''
    rows, cols, bands = references.shape
    d = 1 / ratio  # 全色图像与高光谱图像空间分辨率之比
    pixels = rows * cols
    ref_temp = np.reshape(references, [pixels, bands], order='F')
    tar_temp = np.reshape(target, [pixels, bands], order='F')
    err = ref_temp - tar_temp
    rmse2 = np.sum(err ** 2, axis=0) / pixels  # 均方误差
    uk = np.mean(tar_temp, axis=0)  # 各通道像素均值
    relative_rmse2 = rmse2 / uk ** 2  # 相对均方误差
    total_relative_rmse = np.sum(relative_rmse2)  # 求和
    out = 100 * d * np.sqrt(1 / bands * total_relative_rmse)  # 总体相对误差
    return out


def quality_mesure_fun(target_path,reference_path):
    num_start = 21
    num_end = 32
    ratio = 8
    out = {}
    average_out = {'cc':0,'sam':0,'psnr':0,'rmse':0,'egras':0,'ssim':0}
    for i in range(num_start,num_end + 1):
        mat = sio.loadmat(reference_path+'%d.mat'%i)
        reference = mat['label']
        target = sio.loadmat(target_path + '%d.mat'%i)['hs']
        target = np.float32(target)
        target[target < 0] = 0.0
        target[target > 1] = 1.0
        quality_accessment(out,reference,target,ratio)
        # print(out)
        for key in out.keys():
            average_out[key] += out[key]
        print('image %d has finished'%i)
    for key in average_out.keys():
        average_out[key] /= (num_end-num_start+1)
    print(average_out)




