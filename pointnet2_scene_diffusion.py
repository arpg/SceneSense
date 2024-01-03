import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        #can change the 3 in the 3+3 to change the numner of poitns going into this
        #can change the first number in the abstraction to change the downsampling\
        #originally reduced from 2048 to 1024
        #another example did 
        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, 6 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        # print(l3_points.shape)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.shape)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.shape)
        #i might actually want this instead of the l0 points since it will be much smaller
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        # print(l0_points.shape)

        # x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        #I am going to add the dropout to the l1 points because we dont actuall need that much conditioning
        x = self.drop1(F.relu(self.bn1(self.conv1(l1_points))))
        return x

        #removing this and returning above because we dont need all this data
        # print(x.shape)
        # #this is where they convolve the features into the softmax part
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        # return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))