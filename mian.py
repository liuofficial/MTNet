import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import Network
import time
import os
from utils import PSNR_GPU,cutCAVEPieces_Test,merge_Cave_test,Dataset,checkFile,PSNR,quality_mesure_fun
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class MTNet:
    def __init__(self):
        super().__init__()
        self.itration_K = 1
        self.net = Network.TransBlock(hs_band=31,pan_band=1,itra_num=self.itration_K)
        self.train_data_path = "E:/Datasets/CAVE/CAVE_48_patch/train/"
        self.vaild_data_path = "E:/Datasets/CAVE/CAVE_48_patch/valid/"
        self.batch_size = 16
        self.learning_rate = 2e-4
        self.net_save_path = './Dict/' + 'Iteration_' + str(self.itration_K) + '/'
        #Since the CAVE dataset is too large to directly test,
        # we split the CAVE test set non-overlapping to a size of 128*128 for testing,
        # the other datasets do not need to be performed as above.
        self.data_path = 'E:/DATA/CAVEMAT/'
        self.cutpatch_path = 'E:/Datasets/CAVE/cave_test_patch_128/'

        self.patchtest_path = './Results/CAVE_patch_HRHS/'
        self.results_path = './Results/CAVE_test_HRHS/'
        self.max_epoch = 300

    def load_data(self, path, batch):
        datasets = Dataset(path)
        loader = torch.utils.data.DataLoader(datasets, batch_size=batch, shuffle=True)
        return loader

    def train(self):
        trainloader = self.load_data(self.train_data_path,self.batch_size)
        vaildloader = self.load_data(self.vaild_data_path, self.batch_size)
        self.net.cuda()
        optimizer = optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.criterion = nn.L1Loss()
        max_psnr = 35
        for epoch in range(self.max_epoch):
            for i, data in enumerate(trainloader, 0):
                Y = data['Y'].cuda()
                Z = data['Z'].cuda()
                X = data['X'].cuda()
                optimizer.zero_grad()
                out = self.net(Y, Z)
                loss = self.criterion(out, X)
                psnr = PSNR_GPU(X, out)
                loss.backward()
                optimizer.step()
                if ((i + 1) % 300 == 0):
                    print("Epoch: [%2d/%4d] [%4d/%4d] loss: %.8f,  PSNR: %8f" %
                          ((epoch + 1), self.max_epoch, (i + 1), len(trainloader), loss.data, psnr))
                ### valid
            loss1,psnr1 = self.Validate(vaildloader)
            if (psnr1 > max_psnr):
                print("get a satisfying model,the valid loss: %.8f,PSNR: %8f" %(loss1.data, psnr1))
                max_psnr = psnr1
                checkFile(self.net_save_path)
                torch.save(self.net.state_dict(), self.net_save_path + 'the_CAVE_maxpsnr_model.pkl')

    def Validate(self,validloader):
        self.net.eval()
        for k, data1 in enumerate(validloader, 0):
            Y = data1['Y'].cuda()
            Z = data1['Z'].cuda()
            X = data1['X'].cuda()
            output = self.net(Y, Z)
            loss = self.criterion(output, X)
            psnr = PSNR_GPU(X, output)
            return loss,psnr

    def Test(self):
        net = self.net
        net.load_state_dict(torch.load(self.net_save_path +'the_CAVE_maxpsnr_model.pkl'))
        # As the image size of the CAVE dataset is 512*512. it is too large to test directly,
        # so its test set is cut into 128*128 for testing and then stitched into 512*512 images
        if(os.path.exists(self.patchtest_path)):
            num = len(os.listdir(self.cutpatch_path))
        else:
            num = cutCAVEPieces_Test(self.data_path,self.cutpatch_path)
        stat = 1
        end = num
        run_time = 0
        start = time.perf_counter()
        checkFile(self.patchtest_path)
        for j in range(stat, end + 1):
            path = self.cutpatch_path + str(j) + '.mat'
            data = sio.loadmat(path)
            pan = data['Z']
            lrhs = data['Y']
            label = data['X']
            lrhs = np.transpose(lrhs, (2, 0, 1))
            lrhs = torch.from_numpy(lrhs).type(torch.FloatTensor).unsqueeze(0)
            pan = np.transpose(pan, (2, 0, 1))
            pan = torch.from_numpy(pan).type(torch.FloatTensor).unsqueeze(0)
            out = net(lrhs,pan)
            end = time.perf_counter()
            run_time += end - start
            out1 = out.cpu().squeeze().detach().numpy()
            out1 = np.transpose(out1, (1, 2, 0))
            out1[out1 < 0] = 0.0
            out1[out1 > 1] = 1.0
            psnr = PSNR(label, out1)
            print('%d has finished,PSNR:%.5f' % (j,psnr))
            sio.savemat(self.patchtest_path + str(j) + '.mat', {'hs': out1})
        print(run_time)
        merge_Cave_test(self.patchtest_path,self.results_path)
        quality_mesure_fun(self.results_path,self.data_path)

if __name__ == '__main__':
    mtnet = MTNet()
    mtnet.train()
    mtnet.Test()









