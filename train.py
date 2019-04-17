import argparse
import json
import os
import shutil
import time

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset.CULane import CULane
from model import LaneNet
from utils.tensorboard import TensorBoard
from utils.transforms import *
from utils.lr_scheduler import PolyLR
from utils.postprocess import embedding_post_process

# ------------ config ------------
exp_dir = "./experiments/exp0"

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)

device = torch.device(exp_cfg['device'])
tensorboard = TensorBoard(exp_dir)

# ------------ train data ------------
transform_train = Compose(Resize((800, 288)), Rotation(2), ToTensor(),
                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])) # imagenet mean, std
train_dataset = CULane(CULane_path, "train", transform_train)
train_loader = DataLoader(train_dataset, **exp_cfg['dataset'], shuffle=True, collate_fn=train_dataset.collate, num_workers=8)

# ------------ val data ------------
transform_val = Compose(Resize((800, 288)), ToTensor(),
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
val_dataset = CULane(CULane_path, "val", transform_val)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=train_dataset.collate, num_workers=4)

# ------------ preparation ------------
net = LaneNet(pretrained=True, **exp_cfg['net'])
net = net.to(device)
net = torch.nn.DataParallel(net)

optimizer = optim.SGD(net.parameters(), **exp_cfg['optim'])
lr_scheduler = PolyLR(optimizer, 0.9, exp_cfg['MAX_ITER'])
best_val_loss = 1e6


def train(epoch):
    print("Train Epoch: {}".format(epoch))
    net.train()
    train_loss = 0
    train_loss_bin_seg = 0
    train_loss_var = 0
    train_loss_dist = 0
    train_loss_reg = 0

    progressbar = tqdm(range(len(train_loader)))

    for batch_idx, sample in enumerate(train_loader):
        img = sample['img'].to(device)
        segLabel = sample['segLabel'].to(device)

        optimizer.zero_grad()
        output = net(img, segLabel)
        embedding = output['embedding']
        binary_seg = output['binary_seg']
        seg_loss = output['seg_loss']
        var_loss = output['var_loss']
        dist_loss = output['dist_loss']
        reg_loss = output['reg_loss']
        loss = output['loss']
        if isinstance(net, torch.nn.DataParallel):
            seg_loss = seg_loss.sum()
            var_loss = var_loss.sum()
            dist_loss = dist_loss.sum()
            reg_loss = reg_loss.sum()
            loss = output['loss'].sum()

        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 4:
            lr_scheduler.step()

        train_loss += loss.item()
        train_loss_bin_seg += seg_loss.item()
        train_loss_var += var_loss.item()
        train_loss_dist += dist_loss.item()
        train_loss_reg += reg_loss.item()
        progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
        progressbar.update(1)

    progressbar.close()
    tensorboard.scalar_summary("train_loss", train_loss, epoch)
    tensorboard.scalar_summary("train_loss_bin_seg", train_loss_bin_seg, epoch)
    tensorboard.scalar_summary("train_loss_var", train_loss_var, epoch)
    tensorboard.scalar_summary("train_loss_dist", train_loss_dist, epoch)
    tensorboard.scalar_summary("train_loss_reg", train_loss_reg, epoch)
    tensorboard.writer.flush()

    if epoch % 1 == 0:
        save_dict = {
            "epoch": epoch,
            "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }
        save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth')
        torch.save(save_dict, save_name)
        print("model is saved: {}".format(save_name))

    print("------------------------\n")


def val(epoch):
    global best_val_loss

    print("Val Epoch: {}".format(epoch))

    net.eval()
    val_loss = 0
    val_loss_bin_seg = 0
    val_loss_var = 0
    val_loss_dist = 0
    val_loss_reg = 0
    progressbar = tqdm(range(len(val_loader)))

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            img = sample['img'].to(device)
            segLabel = sample['segLabel'].to(device)

            output = net(img, segLabel)
            embedding = output['embedding']
            binary_seg = output['binary_seg']
            seg_loss = output['seg_loss']
            var_loss = output['var_loss']
            dist_loss = output['dist_loss']
            reg_loss = output['reg_loss']
            loss = output['loss']
            if isinstance(net, torch.nn.DataParallel):
                seg_loss = seg_loss.sum()
                var_loss = var_loss.sum()
                dist_loss = dist_loss.sum()
                reg_loss = reg_loss.sum()
                loss = output['loss'].sum()

            # visualize validation every 5 frame, 50 frames in all
            gap_num = 5
            if batch_idx%gap_num == 0 and batch_idx < 50 * gap_num:
                color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8') # bgr
                display_imgs = []
                embedding = embedding.detach().cpu().numpy()
                bin_seg_prob = binary_seg.detach().cpu().numpy()
                bin_seg_pred = np.argmax(bin_seg_prob, axis=1)

                for b in range(len(img)):
                    img_name = sample['img_name'][b]
                    img = cv2.imread(img_name) # BGR
                    img = cv2.resize(img, (800, 288))

                    bin_seg_img = np.zeros_like(img)
                    bin_seg_img[bin_seg_pred[b]==1] = [0, 0, 255]

                    # # ----------- cluster ---------------
                    # seg_img = np.zeros_like(img)
                    # embedding_b = np.transpose(embedding[b], (1, 2, 0))
                    # lane_seg_img = embedding_post_process(embedding_b, bin_seg_pred[b], exp_cfg['net']['delta_v'])
                    # embed_unique_idxs = np.unique(lane_seg_img)
                    # embed_unique_idxs = embed_unique_idxs[embed_unique_idxs!=0]
                    # for i, lane_idx in enumerate(embed_unique_idxs):
                    #     seg_img[lane_seg_img==lane_idx] = color[i]
                    # img = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=img, beta=1., gamma=0.)

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    bin_seg_img = cv2.cvtColor(bin_seg_img, cv2.COLOR_BGR2RGB)

                    display_imgs.append(img)
                    display_imgs.append(bin_seg_img)

                tensorboard.image_summary("img_{}".format(batch_idx), display_imgs, epoch)

            val_loss += loss.item()
            val_loss_bin_seg += seg_loss.item()
            val_loss_var += var_loss.item()
            val_loss_dist += dist_loss.item()
            val_loss_reg += reg_loss.item()

            progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
            progressbar.update(1)

    progressbar.close()
    tensorboard.scalar_summary("val_loss", val_loss, epoch)
    tensorboard.scalar_summary("val_loss_bin_seg", val_loss_bin_seg, epoch)
    tensorboard.scalar_summary("val_loss_var", val_loss_var, epoch)
    tensorboard.scalar_summary("val_loss_dist", val_loss_dist, epoch)
    tensorboard.scalar_summary("val_loss_reg", val_loss_reg, epoch)
    tensorboard.writer.flush()

    print("------------------------\n")
    if val_loss < best_val_loss:
        save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth')
        copy_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '_best.pth')
        shutil.copyfile(save_name, copy_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.resume:
        save_dict = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth'))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(save_dict['net'])
        else:
            net.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        start_epoch = save_dict['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, 100):
        train(epoch)
        if epoch % 2 == 0:
            print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val(epoch)


if __name__ == "__main__":
    main()
