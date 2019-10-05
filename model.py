import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LaneNet(nn.Module):
    def __init__(
            self,
            embed_dim=4,
            delta_v=0.5,
            delta_d=3.0,
            scale_lane_line=1.0,
            scale_var=1.0,
            scale_dist=1.0,
            pretrained=False,
            **kwargs
    ):
        super(LaneNet, self).__init__()
        self.pretrained = pretrained
        self.embed_dim = embed_dim
        self.delta_v = delta_v
        self.delta_d = delta_d

        self.net_init()

        self.scale_seg = scale_lane_line
        self.scale_var = scale_var
        self.scale_dist = scale_dist
        self.scale_reg = 0
        self.seg_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 1.]))

    def net_init(self):
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features

        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')

        # ----------------- additional conv -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

        # ----------------- embedding -----------------
        self.embedding = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, self.embed_dim, 1)
        )

        # ----------------- binary segmentation -----------------
        self.binary_seg = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 2, 1)
        )

    def forward(self, img, segLabel=None):
        x = self.backbone(img)
        x = self.layer1(x)

        embedding = self.embedding(x)
        binary_seg = self.binary_seg(x)

        if segLabel is not None:
            var_loss, dist_loss, reg_loss = self.discriminative_loss(embedding, segLabel)
            seg_loss = self.seg_loss(binary_seg, torch.gt(segLabel, 0).type(torch.long))
        else:
            var_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
            dist_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
            seg_loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        loss = seg_loss * self.scale_seg + var_loss * self.scale_var + dist_loss * self.scale_dist

        output = {
            "embedding": embedding,
            "binary_seg": binary_seg,
            "loss_seg": seg_loss,
            "loss_var": var_loss,
            "loss_dist": dist_loss,
            "loss": loss
        }

        return output

    def discriminative_loss(self, embedding, seg_gt):
        batch_size = embedding.shape[0]

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b] # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels!=0]
            num_lanes = len(labels)
            if num_lanes==0:
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean( F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim,1), dim=0) - self.delta_v)**2 ) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * self.delta_d  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size
        return var_loss, dist_loss, reg_loss