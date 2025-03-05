import torch.nn as nn
import torch



class Metric():
    def __init__(self, names):
        self.names = names

    def get_metrics(self, pre, true, metrics, n_metrics, lead_time):
        for name in self.names:
            method = getattr(self, name, "Default")
            n, m = method(pre, true)
            n_metrics[name][lead_time] = n_metrics[name][lead_time] + n
            metrics[name][lead_time] = metrics[name][lead_time] + m
        return metrics
    

    def csi_15(self, pre, true):
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>15
        true = true>15
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
       
        d = TP+FN+FP
        u = TP
        if TP+FN > 0:
            csi = u/d
            return 1, csi.item()
        else:
            return 0, 0

    def csi_10(self, pre, true):
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>10
        true = true>10
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN+FP
        u = TP
        if TP+FN > 0:
            csi = u/d
            return 1, csi.item()
        else:
            return 0, 0

    def csi_20(self, pre, true):
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>20
        true = true>20
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN+FP
        u = TP
        if TP+FN > 0:
            csi = u/d
            return 1, csi.item()
        else:
            return 0, 0

                

    def csi_30(self, pre, true):
       
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>30
        true = true>30
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN+FP
        u = TP
        if TP+FN > 0:
            csi = u/d
            return 1, csi.item()
        else:
            return 0, 0

    def csi_40(self, pre, true):
     
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>40
        true = true>40
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN+FP
        u = TP
        if TP+FN > 0:
            csi = u/d
            return 1, csi.item()
        else:
            return 0, 0
        
    def csi_45(self, pre, true):
     
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>45
        true = true>45
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN+FP
        u = TP
        if TP+FN > 0:
            csi = u/d
            return 1, csi.item()
        else:
            return 0, 0
    
    def csi_90(self, pre, true):
     
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>90
        true = true>90
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN+FP
        u = TP
        if TP+FN > 0:
            csi = u/d
            return 1, csi.item()
        else:
            return 0, 0
    
    def csi_135(self, pre, true):
     
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>135
        true = true>135
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN+FP
        u = TP
        if TP+FN > 0:
            csi = u/d
            return 1, csi.item()
        else:
            return 0, 0
        
    def pod_20(self, pre, true):
      
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>20
        true = true>20
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN
        u = TP
        if d > 0:
            pod = u/d
            return 1, pod.item()
        else:
            return 0, 0
        
    def pod_30(self, pre, true):
      
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>30
        true = true>30
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN
        u = TP
        if d > 0:
            pod = u/d
            return 1, pod.item()
        else:
            return 0, 0
    
    def pod_40(self, pre, true):
      
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>40
        true = true>40
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN
        u = TP
        if d > 0:
            pod = u/d
            return 1, pod.item()
        else:
            return 0, 0

    def pod_90(self, pre, true):
      
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>90
        true = true>90
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN
        u = TP
        if d > 0:
            pod = u/d
            return 1, pod.item()
        else:
            return 0, 0
        
    def far_20(self, pre, true):
   
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>20
        true = true>20
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FP
        u = FP
        if d > 0:
            far = u/d
            return 1, far.item()
        else:
            return 0, 0
        
    def far_30(self, pre, true):
   
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>30
        true = true>30
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FP
        u = FP
        if d > 0:
            far = u/d
            return 1, far.item()
        else:
            return 0, 0
        
    def far_45(self, pre, true):
   
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>45
        true = true>45
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FP
        u = FP
        if d > 0:
            far = u/d
            return 1, far.item()
        else:
            return 0, 0
        
    def far_90(self, pre, true):
   
        pre = pre[0, :, :]
        true = true[0, :, :]
        pre = pre>90
        true = true>90
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FP
        u = FP
        if d > 0:
            far = u/d
            return 1, far.item()
        else:
            return 0, 0



    def psnr(self, pre, true):
        mse = nn.functional.mse_loss(pre[0, :, :], true[0, :, :])
        ma = 255
        psnr = 10 * torch.log10(ma ** 2 / mse)
        return 1, psnr.item()

    def ssim(self, pre, true):
        C1 = (0.001 * 255) ** 2
        C2 = (0.003 * 255) ** 2
        img1 = pre[0, :, :]
        img2 = true[0, :, :]
        mean1 = torch.mean(img1)
        mean2 = torch.mean(img2)
        va1 = torch.var(img1)
        va2 = torch.var(img2)
        co = torch.mean((img1 - mean1) * (img2 - mean2))
        ssim = ((2 * mean1 * mean2 + C1) * (2 * co + C2)) / ((mean1 ** 2 + mean2 ** 2 + C1) * (va1 + va2 + C2))
        return 1, ssim.item()

    def mse(self, pre, true):
        mse = nn.functional.mse_loss(pre[0, :, :], true[0, :, :])
        return 1, mse.item()
