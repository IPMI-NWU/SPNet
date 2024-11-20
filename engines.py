import random
import time
import datetime
import torch.nn.functional as Func
import numpy as np
import torch
import torch.nn as nn
import util.misc as utils
from torch.autograd import Variable
import torchvision
from inference import keep_largest_connected_components
from util.Jigsaw import Jigsaw, RandomBrightnessContrast


def get_pseudo_weight(epoch):
    phase = 300
    if epoch > phase:
        return 0.1
    else:
        return 0.001 + 0.099 * np.sin(2 * np.pi * epoch / (phase * 4))


def get_edge_weight(epoch):
    phase = 100
    if epoch > phase:
        return 0
    else:
        return 0.05 * np.cos(2 * np.pi * epoch / (phase * 4))


class pDLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super(pDLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * ignore_mask)
        y_sum = torch.sum(target * target * ignore_mask)
        z_sum = torch.sum(score * score * ignore_mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        ignore_mask = torch.ones_like(target)
        ignore_mask[target == self.ignore_index] = 0
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class Visualize_train(nn.Module):
    def __init__(self):
        super().__init__()

    def save_image(self, image, tag, epoch, writer):
        if tag == 'sample' or tag == 'sample_color' or tag == 'sample_jig_2' or tag == 'sample_jig_4':
            image_max, image_min = 10, -1
            image = (image - image_min) / (image_max - image_min)
            image = torch.clamp(image, 0.0, 1.0)
            grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        else:
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        writer.add_image(tag, grid, epoch)

    def forward(self, sample_list, label_list, output_list, pseudo_label_list,
                sample_jig_2_list, sample_jig_4_list,
                output_jig_temp_2_list, output_jig_temp_4_list,
                output_jig_2_list, output_jig_4_list,
                sample_color_list, output_color_list,
                pseudo_pred_list, edge_list,
                epoch, writer):
        self.save_image(sample_list.float(), 'sample', epoch, writer)
        self.save_image(label_list.float(), 'label', epoch, writer)
        self.save_image(output_list.float(), 'output', epoch, writer)
        self.save_image(pseudo_label_list.float(), 'pseudo_label', epoch, writer)

        self.save_image(sample_jig_2_list.float(), 'sample_jig_2', epoch, writer)
        self.save_image(sample_jig_4_list.float(), 'sample_jig_4', epoch, writer)

        self.save_image(output_jig_temp_2_list.float(), 'output_jig_temp_2', epoch, writer)
        self.save_image(output_jig_temp_4_list.float(), 'output_jig_temp_4', epoch, writer)

        self.save_image(output_jig_2_list.float(), 'output_jig_2', epoch, writer)
        self.save_image(output_jig_4_list.float(), 'output_jig_4', epoch, writer)

        self.save_image(sample_color_list.float(), 'sample_color', epoch, writer)
        self.save_image(output_color_list.float(), 'output_color', epoch, writer)

        self.save_image(pseudo_pred_list.float(), 'pseudo_pred', epoch, writer)
        self.save_image(edge_list.float(), 'edge_list', epoch, writer)


def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    # unique = target_masks.unique()
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot_dim4(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader_dict: dict, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args, writer):
    # ------------------------------------------------------------
    # criterion = partial Cross Entropy loss
    # ------------------------------------------------------------
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    # ------------------------------------------------------------
    # numbers: {'MR': 382/batch_size}
    # iterats: {'MR': DataLoader}
    # counts: {'MR': 1}
    # total_steps: 382/batch_size
    # ------------------------------------------------------------
    numbers = {k: len(v) for k, v in dataloader_dict.items()}
    dataloader = dataloader_dict['MR']
    tasks = dataloader_dict.keys()
    counts = {k: 0 for k in tasks}
    total_steps = sum(numbers.values())
    start_time = time.time()

    sample_list, label_list, output_list, pseudo_label_list = [], [], [], []
    sample_jig_2_list, sample_jig_4_list = [], []
    output_jig_temp_2_list, output_jig_temp_4_list = [], []
    sample_color_list, output_color_list = [], []
    output_jig_2_list, output_jig_4_list = [], []
    pseudo_pred_list, edge_list = [], []
    step = 0
    for sample, label, edge in dataloader:
        start = time.time()
        tasks = [t for t in tasks if counts[t] < numbers[t]]
        task = random.sample(tasks, 1)[0]
        counts.update({task: counts[task] + 1})
        datatime = time.time() - start
        # -------------------------------------------------------
        # samples: {mask: Tensor[bs,212,212], tensors: Tensor[bs,1,212,212]}
        # -------------------------------------------------------
        sample = sample.to(device)
        sample = Variable(sample.tensors, requires_grad=True)
        # -------------------------------------------------------
        # label[0]: {name:'subject7_DE_scribble.nii', slice:[-1,-1,6], masks:tensor(1,212,212), orig_size:[1,480,480]}
        # label: tensor(bs,5,212,212)
        # one-hot形式 0:BG, 1:RV, 2:Myo, 3:LV, 4:unlabeled pixels
        # -------------------------------------------------------
        label = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in label]
        label = convert_targets(label, device)

        edge = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in edge]
        edge = torch.stack([t["masks"] for t in edge]) / 255.0
        # -------------------------------------------------------
        # 加噪
        # -------------------------------------------------------
        adv_mask1, adv_mask2 = np.random.binomial(n=1, p=0.1), np.random.binomial(n=1, p=0.1)
        if adv_mask1 == 1 or adv_mask2 == 1:
            noise = torch.zeros_like(sample).uniform_(-1.0 * 10.0 / 255., 10.0 / 255.)
            sample = Variable(sample + noise, requires_grad=True)
        # -------------------------------------------------------
        # color
        # -------------------------------------------------------
        sample_color = RandomBrightnessContrast(sample, brightness_limit=0.7, contrast_limit=0.7, p=1)
        # -------------------------------------------------------
        # 拼图打乱
        # -------------------------------------------------------
        sample_jig_2, shuffle_index_2 = Jigsaw(sample, 2, 2)
        sample_jig_4, shuffle_index_4 = Jigsaw(sample, 4, 4)
        # -------------------------------------------------------
        # 模型预测
        # -------------------------------------------------------
        output = model(sample, task)
        output_color = model(sample_color, task)
        output_jig_temp_2 = model(sample_jig_2, task)
        output_jig_temp_4 = model(sample_jig_4, task)
        # -------------------------------------------------------
        # 拼图还原
        # -------------------------------------------------------
        output_jig_2, _ = Jigsaw(output_jig_temp_2["pred_masks"], 2, 2, shuffle_index_2)
        output_jig_4, _ = Jigsaw(output_jig_temp_4["pred_masks"], 4, 4, shuffle_index_4)
        output_jig_2 = {'pred_masks': output_jig_2}
        output_jig_4 = {'pred_masks': output_jig_4}
        # -------------------------------------------------------
        # 伪标签生成
        # -------------------------------------------------------
        beta = np.random.rand(4)
        beta = [b / np.sum(beta) for b in beta]
        pseudo_pred = beta[0] * output["pred_masks"] + beta[1] * output_jig_2["pred_masks"] + \
                      beta[2] * output_jig_4["pred_masks"] + \
                      beta[3] * output_color["pred_masks"]
        pseudo_label = torch.argmax(pseudo_pred.detach(), dim=1, keepdim=False).unsqueeze(1)
        # -------------------------------------------------------
        # integrity loss
        # -------------------------------------------------------
        pred = output["pred_masks"]
        predictions_original_list = []
        for i in range(pred.shape[0]):
            prediction = np.uint8(np.argmax(pred[i, :, :, :].detach().cpu(), axis=0))
            prediction = keep_largest_connected_components(prediction)
            prediction = torch.from_numpy(prediction).to(device)
            predictions_original_list.append(prediction)
        predictions = torch.stack(predictions_original_list)
        predictions = torch.unsqueeze(predictions, 1)
        pred_keep_largest_connected = to_onehot_dim4(predictions, device)
        # -------------------------------------------------------
        # loss
        # -------------------------------------------------------
        weight_dict = criterion.weight_dict

        loss_dict = criterion(output, label)
        loss_scribble = sum(loss_dict[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        loss_scribble = 1.0 * loss_scribble

        loss_scribble_jig_2 = criterion(output_jig_2, label)
        loss_scribble_jig_2 = sum(
            loss_scribble_jig_2[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        loss_scribble_jig_2 = 1.0 * loss_scribble_jig_2

        loss_scribble_jig_4 = criterion(output_jig_4, label)
        loss_scribble_jig_4 = sum(
            loss_scribble_jig_4[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        loss_scribble_jig_4 = 0.1 * loss_scribble_jig_4

        loss_scribble_color = criterion(output_color, label)
        loss_scribble_color = sum(
            loss_scribble_color[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        loss_scribble_color = 0.2 * loss_scribble_color

        loss_consistency_1_2 = 1 - Func.cosine_similarity(output_jig_2["pred_masks"], output["pred_masks"],
                                                          dim=1).mean()
        loss_consistency_1_2 = 0.3 * loss_consistency_1_2

        loss_consistency_1_4 = 1 - Func.cosine_similarity(output_jig_4["pred_masks"], output["pred_masks"],
                                                          dim=1).mean()
        loss_consistency_1_4 = 0.1 * loss_consistency_1_4

        loss_consistency_1_c = 1 - Func.cosine_similarity(output_color["pred_masks"], output["pred_masks"],
                                                          dim=1).mean()
        loss_consistency_1_c = 0.1 * loss_consistency_1_c

        loss_integrity = 1 - Func.cosine_similarity(pred[:, 0:4, :, :], pred_keep_largest_connected, dim=1).mean()
        loss_integrity = 0.3 * loss_integrity
        # -------------------------------------------------------
        # pseudo loss
        # -------------------------------------------------------
        dice_loss = pDLoss(4, ignore_index=4)
        loss_pseudo = get_pseudo_weight(epoch) * (dice_loss(output["pred_masks"], pseudo_label) +
                                                  dice_loss(output_jig_2["pred_masks"], pseudo_label) +
                                                  1.0 * dice_loss(output_jig_4["pred_masks"], pseudo_label) +
                                                  1.0 * dice_loss(output_color["pred_masks"], pseudo_label))
        # -------------------------------------------------------
        # edge loss
        # -------------------------------------------------------
        pseudo_pred = pseudo_pred[:, 1:, :]
        pseudo_pred = torch.sum(pseudo_pred, dim=1, keepdim=True)
        mse_loss = nn.MSELoss()
        loss_edge = mse_loss(pseudo_pred, edge)
        loss_edge = get_edge_weight(epoch) * loss_edge
        # -------------------------------------------------------
        # LOSS
        # -------------------------------------------------------
        loss = loss_scribble + loss_scribble_jig_2 + loss_scribble_jig_4 + loss_scribble_color + \
               loss_consistency_1_2 + loss_consistency_1_4 + loss_consistency_1_c + \
               loss_integrity + loss_pseudo + loss_edge
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # -------------------------------------------------------
        # 可视化
        # -------------------------------------------------------
        if step == 0:
            sample_list.append(sample[0].detach())
            label_list.append(label.argmax(1, keepdim=True)[0].detach())
            output_list.append(output['pred_masks'].argmax(1, keepdim=True)[0].detach())
            pseudo_label_list.append(pseudo_label[0].detach())

            sample_color_list.append(sample_color[0].detach())
            output_color_list.append(output_color['pred_masks'].argmax(1, keepdim=True)[0].detach())

            sample_jig_2_list.append(sample_jig_2[0].detach())
            output_jig_temp_2_list.append(output_jig_temp_2['pred_masks'].argmax(1, keepdim=True)[0].detach())
            output_jig_2_list.append(output_jig_2['pred_masks'].argmax(1, keepdim=True)[0].detach())

            sample_jig_4_list.append(sample_jig_4[0].detach())
            output_jig_temp_4_list.append(output_jig_temp_4['pred_masks'].argmax(1, keepdim=True)[0].detach())
            output_jig_4_list.append(output_jig_4['pred_masks'].argmax(1, keepdim=True)[0].detach())

            pseudo_pred_list.append(pseudo_pred[0].detach())
            edge_list.append(edge[0].detach())
        # -------------------------------------------------------
        # loss
        # -------------------------------------------------------
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss)
        metric_logger.update(loss_scribble=loss_scribble)
        metric_logger.update(loss_scribble_color=loss_scribble_color)
        metric_logger.update(loss_scribble_jig_2=loss_scribble_jig_2)
        metric_logger.update(loss_consistency_1_2=loss_consistency_1_2)
        metric_logger.update(loss_scribble_jig_4=loss_scribble_jig_4)
        metric_logger.update(loss_consistency_1_4=loss_consistency_1_4)
        metric_logger.update(loss_consistency_1_c=loss_consistency_1_c)
        metric_logger.update(loss_pseudo=loss_pseudo)
        metric_logger.update(loss_integrity=loss_integrity)
        metric_logger.update(loss_edge=loss_edge)

        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        step = step + 1
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    visual_train = Visualize_train()

    writer.add_scalar('loss', loss.item(), epoch)
    writer.add_scalar('loss_scribble', loss_scribble.item(), epoch)
    writer.add_scalar('loss_scribble_jig_2', loss_scribble_jig_2.item(), epoch)
    writer.add_scalar('loss_scribble_color', loss_scribble_color.item(), epoch)
    writer.add_scalar('loss_consistency_1_2', loss_consistency_1_2.item(), epoch)
    writer.add_scalar('loss_scribble_jig_4', loss_scribble_jig_4.item(), epoch)
    writer.add_scalar('loss_consistency_1_4', loss_consistency_1_4.item(), epoch)
    writer.add_scalar('loss_consistency_1_c', loss_consistency_1_c.item(), epoch)
    writer.add_scalar('loss_pseudo', loss_pseudo.item(), epoch)
    writer.add_scalar('loss_integrity', loss_integrity.item(), epoch)
    writer.add_scalar('loss_edge', loss_edge.item(), epoch)
    writer.add_scalar('pseudo_weight', get_pseudo_weight(epoch), epoch)

    visual_train(torch.stack(sample_list), torch.stack(label_list), torch.stack(output_list),
                 torch.stack(pseudo_label_list),
                 torch.stack(sample_jig_2_list), torch.stack(sample_jig_4_list),
                 torch.stack(output_jig_temp_2_list), torch.stack(output_jig_temp_4_list),
                 torch.stack(output_jig_2_list), torch.stack(output_jig_4_list),
                 torch.stack(sample_color_list), torch.stack(output_color_list),
                 torch.stack(pseudo_pred_list), torch.stack(edge_list),
                 epoch, writer)
    return stats

