import argparse
import cv2
import torch

from model import LaneNet
from utils.transforms import *
from utils.postprocess import embedding_post_process

net = LaneNet(pretrained=False, embed_dim=7, delta_v=.5, delta_d=3.)
transform = Compose(Resize((800, 288)), ToTensor(),
                    Normalize(mean=(0.3598, 0.3653, 0.3662), std=(0.2573, 0.2663, 0.2756)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, default="demo/demo.jpg", help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str, help="Path to model weights")
    parser.add_argument("--delta_v", '-dv', type=float, default=0.5, help="Value of delta_v")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_path = args.img_path
    weight_path = args.weight_path

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB for net model input
    x = transform(img)[0]
    x.unsqueeze_(0)

    save_dict = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(save_dict['net'])
    net.eval()

    output = net(img)
    embedding = output['embedding']
    embedding = embedding.detach().cpu().numpy()
    embedding = np.transpose(embedding[0], (1, 2, 0))
    binary_seg = output['binary_seg']
    bin_seg_prob = binary_seg.detach().cpu().numpy()
    bin_seg_pred = np.argmax(bin_seg_prob, axis=1)[0]

    seg_img = np.zeros_like(img)
    lane_seg_img = embedding_post_process(embedding, bin_seg_pred, args.delta_v)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    for i, lane_idx in enumerate(np.unique(lane_seg_img)):
        seg_img[lane_seg_img == lane_idx] = color[i]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (800, 288))
    img = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=img, beta=1., gamma=0.)

    cv2.imwrite("demo/demo_result.jpg", img)

    if args.visualize:
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
