"""
FastGAN models adapted for im_size=64 (BloodMNIST native).
Generator64: feat_4 -> feat_8 -> feat_16 -> feat_32 -> feat_64 (with SE at 64).
            outputs [img_64, img_32 (aux)]
Discriminator64: big branch on 64x64 -> 8x8 nfc[16] (with SE_2_16).
                 small branch on 32x32 -> 4x4 nfc[16] for rf_small.
                 reconstruction heads identical to original (decoder_big/small/part output 128x128).
Reuses primitives from models.py (UpBlock, UpBlockComp, DownBlock, DownBlockComp, SEBlock,
                                   InitLayer, SimpleDecoder, weights_init).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import (
    UpBlock, UpBlockComp, DownBlock, DownBlockComp,
    SEBlock, InitLayer, SimpleDecoder,
    conv2d, weights_init,
)


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=64):
        super().__init__()
        assert im_size == 64, "Generator64 only supports im_size=64"
        self.im_size = 64

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2}
        nfc = {k: int(v * ngf) for k, v in nfc_multi.items()}

        self.init     = InitLayer(nz, channel=nfc[4])
        self.feat_8   = UpBlockComp(nfc[4],  nfc[8])
        self.feat_16  = UpBlock(    nfc[8],  nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32])
        self.feat_64  = UpBlock(    nfc[32], nfc[64])

        self.se_64    = SEBlock(nfc[4], nfc[64])

        self.to_big   = conv2d(nfc[64], nc, 3, 1, 1, bias=False)
        self.to_aux   = conv2d(nfc[32], nc, 1, 1, 0, bias=False)

    def forward(self, noise):
        feat_4  = self.init(noise)
        feat_8  = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))

        im_64 = torch.tanh(self.to_big(feat_64))
        im_32 = torch.tanh(self.to_aux(feat_32))
        return [im_64, im_32]


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=64):
        super().__init__()
        assert im_size == 64, "Discriminator64 only supports im_size=64"
        self.im_size = 64
        self.ndf = ndf

        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5}
        nfc = {k: int(v * ndf) for k, v in nfc_multi.items()}

        self.down_from_big = nn.Sequential(
            conv2d(nc, nfc[128], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down_4  = DownBlockComp(nfc[128], nfc[64])
        self.down_8  = DownBlockComp(nfc[64],  nfc[32])
        self.down_16 = DownBlockComp(nfc[32],  nfc[16])

        self.se_2_16 = SEBlock(nfc[128], nfc[16])

        self.rf_big = nn.Sequential(
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            nn.BatchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False),
        )

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[64], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[64], nfc[32]),
            DownBlock(nfc[32], nfc[16]),
        )
        self.rf_small = conv2d(nfc[16], 1, 4, 1, 0, bias=False)

        self.decoder_big   = SimpleDecoder(nfc[16], nc)
        self.decoder_small = SimpleDecoder(nfc[16], nc)
        self.decoder_part  = SimpleDecoder(nfc[32], nc)

    def forward(self, imgs, label, part=None):
        if not isinstance(imgs, list):
            imgs = [F.interpolate(imgs, size=64), F.interpolate(imgs, size=32)]

        feat_2  = self.down_from_big(imgs[0])
        feat_4  = self.down_4(feat_2)
        feat_8  = self.down_8(feat_4)
        feat_16 = self.se_2_16(feat_2, self.down_16(feat_8))

        rf_0 = self.rf_big(feat_16).view(-1)

        feat_small = self.down_from_small(imgs[1])
        rf_1 = self.rf_small(feat_small).view(-1)

        if label == 'real':
            rec_img_big   = self.decoder_big(feat_16)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            if part == 0:
                rec_img_part = self.decoder_part(feat_8[:, :, :8, :8])
            elif part == 1:
                rec_img_part = self.decoder_part(feat_8[:, :, :8, 8:])
            elif part == 2:
                rec_img_part = self.decoder_part(feat_8[:, :, 8:, :8])
            else:
                rec_img_part = self.decoder_part(feat_8[:, :, 8:, 8:])
            return torch.cat([rf_0, rf_1]), [rec_img_big, rec_img_small, rec_img_part]

        return torch.cat([rf_0, rf_1])


if __name__ == "__main__":
    # Smoke: forward shapes
    G = Generator(im_size=64); G.apply(weights_init); G.cuda()
    D = Discriminator(im_size=64); D.apply(weights_init); D.cuda()
    z = torch.randn(2, 100, device='cuda')
    fake = G(z)
    print(f"G fake: {fake[0].shape} (big) + {fake[1].shape} (aux)")
    pred_fake = D(fake, "fake")
    print(f"D(fake, 'fake'): {pred_fake.shape}")
    real = torch.randn(2, 3, 64, 64, device='cuda')
    pred_real, recs = D(real, "real", part=0)
    print(f"D(real, 'real'): pred={pred_real.shape} recs={[r.shape for r in recs]}")
