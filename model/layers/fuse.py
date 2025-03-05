import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ProjBlock, self).__init__()
        self.one_conv = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))
        self.double_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        )

    def forward(self, x):
        x1 = self.one_conv(x)
        x2 = self.double_conv(x)
        output = x1 + x2 + x
        return output


class GenBlock(nn.Module):
    def __init__(self, fin, fout, ic, use_se=False, dilation=1, double_conv=True):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.double_conv = double_conv
        self.pad = nn.ReflectionPad2d(dilation)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0, dilation=dilation)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        self.norm_0 = SPADE(fin, ic)
        self.norm_1 = SPADE(fmiddle, ic)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, ic)

    def forward(self, x, evo):
        x_s = self.shortcut(x, evo)
        dx = self.conv_0(self.pad(self.actvn(self.norm_0(x, evo))))
        if self.double_conv:
            dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, evo))))

        out = x_s + dx

        return out

    def shortcut(self, x, evo):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, evo))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):  # 将x的均值和方差替换为evo的均值和方差
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        ks = 3

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 64
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=0),
            nn.ReLU()
        )
        self.pad = nn.ReflectionPad2d(pw)
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)

    def forward(self, x, evo):
        normalized = self.param_free_norm(x)
        
        evo = F.adaptive_avg_pool2d(evo, output_size=x.size()[2:])
        
        actv = self.mlp_shared(evo)

        gamma = self.mlp_gamma(self.pad(actv))
        beta = self.mlp_beta(self.pad(actv))
        
        out = normalized * (1 + gamma) + beta
        
        return out


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class spectral_norm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(spectral_norm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
