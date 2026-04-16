
import argparse
import torch
import os
import glob
from tensorboardX import SummaryWriter
import sys
from dataset import find_dataset_def
import torch.backends.cudnn as cudnn
from networks.casmvs import CascadeMVSNet
from networks.ucs import UCSNet
# from networks.casred import Infer_CascadeREDNet
from networks.stsat import ST_SatMVS, Infer_CascadeREDNet
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from tools.utils import *
from dataset.data_io import save_pfm
import matplotlib.pyplot as plt
from tqdm import tqdm

cudnn.benchmark = True


parser = argparse.ArgumentParser(description='A PyTorch Implementation')
parser.add_argument('--model', default="SAMsat", help='select model', choices=['SAMsat', 'red', "casmvs", "ucs"])
parser.add_argument('--geo_model', default="rpc", help='select dataset', choices=["rpc", "pinhole"])
parser.add_argument('--use_qc', default=False, help="whether to use Quaternary Cubic Form for RPC warping.")
parser.add_argument('--dataset_root', default='/remote-home/Cs_ai_qj_new/chenziyang/MVS/MVSrs/open_dataset_rpc/test', help='dataset root')

parser.add_argument('--loadckpt', default="./checkpoints/samsat/rpc/model_000012.ckpt",
                    help='load a specific checkpoint')
# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images.')
parser.add_argument('--ref_view', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')

# Cascade parameters
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--min_interval', type=float, default=2.5, help='min_interval in the bottom stage')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--lamb', type=float, default=1.5, help="lamb in ucs-net")
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--gpu_id', type=str, default="2")
parser.add_argument('--use_tqdm', action='store_true', help='use tqdm progress bar instead of plain log output')
parser.add_argument('--summary_freq_samples', type=int, default=50, help='tensorboard summary frequency in prediction (unit: samples)')

# parse arguments and check
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# print(args.geo_model)
# print(args.dataset_root)
# assert args.geo_model in args.dataset_root, Exception("set the wrong data root")
# assert args.geo_model in args.loadckpt, Exception("set the wrong checkpoint")
# assert args.model in args.loadckpt, Exception("set the wrong checkpoint")

def _clear_tensorboard_events(tb_log_dir):
    event_paths = sorted(glob.glob(os.path.join(tb_log_dir, "events.out.tfevents*")))
    removed = 0
    for event_path in event_paths:
        if os.path.isfile(event_path):
            os.remove(event_path)
            removed += 1
    print(f"cleared {removed} old tensorboard event file(s) in {tb_log_dir}")

def predict():
    print("argv:", sys.argv[1:])
    print_args(args)
    if args.summary_freq_samples <= 0:
        raise ValueError("--summary_freq_samples must be > 0")

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.geo_model)
    # pre_dataset = MVSDataset(args.dataset_root, "pred", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)
    pre_dataset = MVSDataset(args.dataset_root, "test", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)

    Pre_ImgLoader = DataLoader(pre_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    if args.model == "casmvs":
        model = CascadeMVSNet(min_interval=args.min_interval,
                              ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                              depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                              cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                              geo_model=args.geo_model, use_qc=args.use_qc)
        print("===============> Model: Cascade MVS Net ===========>")
    elif args.model == "ucs":
        model = UCSNet(lamb=args.lamb, stage_configs=[int(nd) for nd in args.ndepths.split(",") if nd],
                       base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                       geo_model=args.geo_model, use_qc=args.use_qc)
        print("===============> Model: UCS-Net ===========>")
    elif args.model == "red":
        model = Infer_CascadeREDNet(min_interval=args.min_interval,
                                    ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                                    depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                                    cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                                    geo_model=args.geo_model, use_qc=args.use_qc)
        print("===============> Model: Cascade RED Net ===========>")
    elif args.model == "SAMsat":
        model = ST_SatMVS(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
        print("===============> Model: Our network ===========>")
    else:
        raise Exception("{}? Not implemented yet!".format(args.model))

    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint file specified by args.loadckpt
    loadckpt = args.loadckpt
    if os.path.isdir(loadckpt):
        saved_models = [file for file in os.listdir(loadckpt) if file.endswith(".ckpt")]
        if len(saved_models) == 0:
            raise ValueError("No .ckpt file found in --loadckpt directory")
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
        loadckpt = os.path.join(loadckpt, saved_models[-1])

    if not os.path.isfile(loadckpt):
        raise ValueError("--loadckpt must be a .ckpt file or a directory containing .ckpt")

    print("loading model {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # setup tensorboard log dir
    cur_log_dir = os.path.dirname(loadckpt)
    pred_log_dir = os.path.join(cur_log_dir, 'predict')
    os.makedirs(pred_log_dir, exist_ok=True)
    print(f"log directory: {pred_log_dir}")

    tb_log_dir = os.path.join(pred_log_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    _clear_tensorboard_events(tb_log_dir)
    logger = SummaryWriter(tb_log_dir)
    print(f"tensorboard directory: {tb_log_dir}")

    avg_test_scalars = DictAverageMeter()
    interval_scalars = DictAverageMeter()
    total_time = 0
    processed_samples = 0
    next_log_step = args.summary_freq_samples

    if args.use_tqdm:
        pred_iter = tqdm(enumerate(Pre_ImgLoader), total=len(Pre_ImgLoader), desc="Predicting")
    else:
        pred_iter = enumerate(Pre_ImgLoader)

    for batch_idx, sample in pred_iter:
        start_time = time.time()
        scalar_outputs, image_outputs = predict_sample(model, sample)
        batch_size_cur = int(sample["imgs"].shape[0])
        avg_test_scalars.update(scalar_outputs, weight=batch_size_cur)
        scalar_outputs = {k: float("{0:.6f}".format(v)) for k, v in scalar_outputs.items()}
        interval_scalars.update(scalar_outputs, weight=batch_size_cur)
        total_time += time.time() - start_time

        processed_samples += batch_size_cur
        while processed_samples >= next_log_step:
            interval_metrics = interval_scalars.mean()
            if len(interval_metrics) > 0:
                save_scalars(logger, 'predict', interval_metrics, next_log_step)
            save_images(logger, 'predict', image_outputs, next_log_step)
            interval_scalars = DictAverageMeter()
            next_log_step += args.summary_freq_samples

        first_name = str(sample['out_name'][0])
        if args.use_tqdm:
            pred_iter.set_postfix({'name': first_name, 'time': f'{time.time() - start_time:.3f}'})
        else:
            print("Iter {}/{}, {}, time = {:3f}, test results = {}".format(batch_idx, len(Pre_ImgLoader), first_name, time.time() - start_time, scalar_outputs))

        # save per-sample results (depth_est/confidence + color)
        depth_est_batch = tensor2numpy(image_outputs["depth_est"])
        prob_batch = tensor2numpy(image_outputs["photometric_confidence"])
        batch_size = depth_est_batch.shape[0]
        for i in range(batch_size):
            depth_est = np.squeeze(depth_est_batch[i])
            prob = np.float32(np.squeeze(prob_batch[i]))
            if depth_est.ndim != 2 or prob.ndim != 2:
                raise ValueError(
                    "Predict output shape error at batch {}, sample {}: depth_est={}, prob={}, expected 2D maps.".format(
                        batch_idx, i, depth_est.shape, prob.shape
                    )
                )

            curr_b_view = str(sample['out_view'][i])
            curr_b_name = str(sample['out_name'][i])

            depth_dir = os.path.join(pred_log_dir, "depth_est", curr_b_view)
            conf_dir = os.path.join(pred_log_dir, "confidence", curr_b_view)
            depth_color_dir = os.path.join(depth_dir, "color")
            conf_color_dir = os.path.join(conf_dir, "color")
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(conf_dir, exist_ok=True)
            os.makedirs(depth_color_dir, exist_ok=True)
            os.makedirs(conf_color_dir, exist_ok=True)

            depth_path = os.path.join(depth_dir, "{}.pfm".format(curr_b_name))
            conf_path = os.path.join(conf_dir, "{}.pfm".format(curr_b_name))
            depth_color_path = os.path.join(depth_color_dir, "{}.png".format(curr_b_name))
            conf_color_path = os.path.join(conf_color_dir, "{}.png".format(curr_b_name))

            save_pfm(depth_path, depth_est.astype(np.float32))
            save_pfm(conf_path, prob.astype(np.float32))
            plt.imsave(depth_color_path, depth_est, format='png')
            plt.imsave(conf_color_path, prob, format='png')

        del scalar_outputs, image_outputs

    print("final, time = {:3f}, test results = {}".format(total_time, avg_test_scalars.mean()))

    # log overall (full-dataset average) metrics to tensorboard
    overall_step = processed_samples
    overall_metrics = avg_test_scalars.mean()
    if len(overall_metrics) > 0:
        save_scalars(logger, 'overall', overall_metrics, overall_step)

    # close tensorboard logger
    logger.close()

    record_path = os.path.join(pred_log_dir, 'predict_record.txt')
    with open(record_path, "a+", encoding="utf-8") as f:
        f.write("Predict Metrics: {}\n".format(overall_metrics))


@make_nograd_func
def predict_sample(model, sample):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["cam_para"], sample_cuda["depth_values"])
    depth_est = outputs["stage3"]["depth"]
    # depth_est = outputs["stage3"]["depth_filtered"]
    photometric_confidence = outputs["stage3"]["photometric_confidence"]

    image_outputs = {
        "depth_est": depth_est,
        "photometric_confidence": photometric_confidence,
        "depth_gt": depth_gt,
        "ref_image": sample_cuda["imgs"][:, 0],
        "mask": mask,
        "errormap": (depth_est - depth_gt).abs()
    }

    scalar_outputs = {}

    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
    scalar_outputs["MAE"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
    scalar_outputs["RMSE"] = RMSE_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
    scalar_outputs["threshold_1.0m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1.0)
    scalar_outputs["threshold_2.5m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2.5)
    scalar_outputs["threshold_7.5m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 7.5)

    return tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    predict()
