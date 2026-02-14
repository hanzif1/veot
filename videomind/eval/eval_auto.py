# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import re
import nncore
import torch
from nncore.ops import temporal_area, temporal_intersection, temporal_iof, temporal_iou
from tabulate import tabulate


class SafeInt(int):
    def __truediv__(self, other):
        try:
            return SafeInt(super().__truediv__(other))
        except ZeroDivisionError:
            return SafeInt(0)


def check_ans(options, ans, response):
    if response is None: return False
    
    # === 1. 处理 Ground Truth (ans) ===
    # 统一转为字符串并去空格
    ans_str = str(ans).strip()
    
    # 【核心修复】：如果 ans 是数字字符串 ('0', '1'...)，转为字母 ('a', 'b'...)
    if ans_str.isdigit():
        ans_norm = chr(ord('a') + int(ans_str))
    # 如果是字母，直接转小写
    elif len(ans_str) == 1:
        ans_norm = ans_str.lower()
    else:
        # 如果 ans 是长文本 (比如 "Yellow")，尝试在 options 里反向寻找索引
        ans_norm = ans_str.lower()
        if options:
            for idx, opt in enumerate(options):
                # 如果标准答案包含在选项文本中，或者与选项完全一致
                if ans_str.lower() == str(opt).lower(): 
                    ans_norm = chr(ord('a') + idx)
                    break
    
    # === 2. 处理 Model Prediction (response) ===
    pred_str = str(response).strip()
    if len(pred_str) == 0: return False
    
    # 提取预测结果的第一个字符
    # 处理 "A.", "(A)", "Option A" 等情况
    first_token = pred_str.split(' ')[0].strip("().")
    
    if len(first_token) > 0:
        # 如果模型偶尔输出了数字 '0', '1'，也转为字母
        if first_token.isdigit():
            pred_norm = chr(ord('a') + int(first_token))
        else:
            pred_norm = first_token[0].lower()
    else:
        return False

    # === 3. 比对 ===
    return ans_norm == pred_norm


def compute_iou(pred, span, conf, cgbench_mode, conf_thr):
    pred_tensor = torch.Tensor(pred)
    span_tensor = torch.Tensor(span)

    if cgbench_mode:
        if conf_thr > 0:
            conf_tensor = torch.Tensor(conf)
            keep = torch.cat((torch.LongTensor([0]), torch.where(conf_tensor > conf_thr)[0])).unique()
            pred_tensor = pred_tensor[keep]
        else:
            pred_tensor = pred_tensor[:1]
        pred_area = temporal_area(pred_tensor).sum()
        span_area = temporal_area(span_tensor).sum()
        inter = temporal_intersection(pred_tensor, span_tensor).sum()
        iou = (inter / (pred_area + span_area - inter)).unsqueeze(0)
        assert iou.numel() == 1
    else:
        # 标准 Temporal IoU
        iou = temporal_iou(pred_tensor, span_tensor)

    iou = torch.where(iou.isfinite(), iou, 0)
    return iou


def parse_grounding(response, duration):
    """ 将字符串响应解析为时间片段 [[start, end]] """
    if not response or not isinstance(response, str):
        return None
    
    # 提取所有数字
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", response)]
    
    # 情况1: [start, end] 两个数字，假设是归一化时间
    if len(nums) == 2:
        s, e = nums[0], nums[1]
        # 如果是归一化的 (0~1)，乘 duration
        if s <= 1.0 and e <= 1.0 and duration > 1.0:
            return [[s * duration, e * duration]]
        return [[s, e]]
    
    # 情况2: [ymin, xmin, ymax, xmax] 4个数字 -> 这是空间坐标，不是时间！
    # 如果数据集是做 Temporal Grounding (时间定位)，空间坐标无法计算 Temporal IoU
    # 这里我们只能跳过，或者如果用户确实想算时间，可能逻辑不对
    if len(nums) == 4:
        # print("Warning: Found 4 coords (Spatial), but task requires Temporal IoU. Skipping grounding.")
        return None

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path')
    parser.add_argument('--dataset')
    parser.add_argument('--out_name', default='metrics.log')
    parser.add_argument('--conf_thr', type=float, default=-1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # 支持文件或目录
    if nncore.is_file(args.pred_path):
        pred_paths = [args.pred_path]
    else:
        pred_paths = nncore.ls(args.pred_path, ext=['json', 'jsonl'], join_path=True)
    
    log_file = nncore.join(args.pred_path if nncore.is_dir(args.pred_path) else nncore.dir_name(args.pred_path), args.out_name)
    nncore.set_default_logger(logger='eval', fmt=None, log_file=log_file)

    if args.dataset is not None:
        cgbench_mode = args.dataset == 'cgbench'
        nncore.log(f'CG-Bench mode: {cgbench_mode}')
    else:
        cgbench_mode = False
        nncore.log('Dataset is unknown, using default mode', log_level='WARNING')

    nncore.log(f'Total number of files: {len(pred_paths)}')

    if cgbench_mode:
        top_k = [1]
        thres = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        top_k = [1, 3, 5]
        thres = [0.3, 0.5, 0.7]

    tab_iou, tab_iop, tab_ans = dict(), dict(), dict()
    iou_raise, iou_lower, iop_raise, iop_lower = SafeInt(0), SafeInt(0), SafeInt(0), SafeInt(0)
    tab_iou_all = [SafeInt(0) for _ in range(len(top_k) * len(thres) + 3)]
    tab_iop_all = [SafeInt(0) for _ in range(len(top_k) * len(thres) + 3)]
    tab_ans_all = [SafeInt(0) for _ in range(len(thres) + 5)]

    for path in pred_paths:
        data = nncore.load(path)
        if not isinstance(data, list): data = [data] # 兼容单条json

        for sample in data:
            # === 核心修改 1: 字段映射 (Adapter) ===
            # 将 infer_agent_api.py 的输出映射到 eval_auto.py 期望的格式
            
            # 1. 映射 QA 答案
            if 'model_pred' in sample and 'response' not in sample:
                sample['response'] = sample['model_pred']
            
            # 2. 映射 Grounding 预测
            if 'pred' not in sample and 'grounder_response' in sample:
                duration = sample.get('duration', 10.0)
                parsed_pred = parse_grounding(sample['grounder_response'], duration)
                if parsed_pred:
                    sample['pred'] = parsed_pred
                    sample['conf'] = [1.0] # 默认置信度

            # === 结束映射 ===

            task = sample.get('task', 'unknown')
            if isinstance(task, str):
                task = [task]

            for t in task:
                if t not in tab_iou:
                    tab_iou[t] = [SafeInt(0) for _ in range(len(top_k) * len(thres) + 3)]
                if t not in tab_iop:
                    tab_iop[t] = [SafeInt(0) for _ in range(len(top_k) * len(thres) + 3)]
                if t not in tab_ans:
                    tab_ans[t] = [SafeInt(0) for _ in range(len(thres) + 5)]

            iou_hit = [False for _ in range(len(thres) + 1)]
            iop_hit = False

            # Grounding Eval (只有当存在 pred 和 span 时才计算)
            if 'pred' in sample and 'conf' in sample and 'span' in sample:
                for t in task:
                    tab_iou[t][0] += 1
                    tab_iop[t][0] += 1
                tab_iou_all[0] += 1
                tab_iop_all[0] += 1

                iou = compute_iou(sample['pred'], sample['span'], sample['conf'], cgbench_mode, args.conf_thr)
                
                # 安全检查，防止 iou 为空
                if iou.numel() == 0: 
                    top = 0.0
                else:
                    top = iou[0].max().item()

                for t in task:
                    tab_iou[t][-1] += top
                tab_iou_all[-1] += top

                for i, k in enumerate(top_k):
                    for j, h in enumerate(thres):
                        if iou[:k].max() >= h:
                            for t in task:
                                tab_iou[t][i * len(thres) + j + 2] += 1
                            tab_iou_all[i * len(thres) + j + 2] += 1
                            if k == 1:
                                iou_hit[j + 1] = True
                                if h == 0.5:
                                    iou_hit[0] = True
                
                # IoP 计算
                iop = temporal_iof(torch.Tensor(sample['pred']), torch.Tensor(sample['span']))
                iop = torch.where(iop.isfinite(), iop, 0)
                if iop.numel() == 0:
                    top = 0.0
                else:
                    top = iop[0].max().item()
                
                for t in task:
                    tab_iop[t][-1] += top
                tab_iop_all[-1] += top

                for i, k in enumerate(top_k):
                    for j, h in enumerate(thres):
                        if iop[:k].max() >= h:
                            for t in task:
                                tab_iop[t][i * len(thres) + j + 2] += 1
                            tab_iop_all[i * len(thres) + j + 2] += 1
                            if k == 1 and h == 0.5:
                                iop_hit = True

                # 失败计数
                if not sample.get('grounder_success', True):
                    for t in task:
                        tab_iou[t][1] += 1
                        tab_iop[t][1] += 1
                    tab_iou_all[1] += 1
                    tab_iop_all[1] += 1

            # QA Eval
            if 'question' in sample and 'response' in sample:
                for t in task:
                    tab_ans[t][0] += 1
                tab_ans_all[0] += 1

                correct = check_ans(sample.get('options', []), sample['ans'], sample['response'])

                if correct:
                    for t in task:
                        tab_ans[t][2] += 1
                    tab_ans_all[2] += 1
                    if iou_hit[0]:
                        for t in task:
                            tab_ans[t][3] += 1
                        tab_ans_all[3] += 1
                    if iop_hit:
                        for t in task:
                            tab_ans[t][4] += 1
                        tab_ans_all[4] += 1
                    for i in range(1, len(iou_hit)):
                        if iou_hit[i]:
                            for t in task:
                                tab_ans[t][i + 4] += 1
                            tab_ans_all[i + 4] += 1
                elif correct is None:
                    for t in task:
                        tab_ans[t][1] += 1
                    tab_ans_all[1] += 1

    tasks = sorted(list(set(list(tab_iou.keys()) + list(tab_iop.keys()) + list(tab_ans.keys()))))

    # --- Print Reports ---
    # 只有当有数据时才打印，防止报错
    
    if tab_iou_all[0] > 0:
        nncore.log('\nGrounding (IoU):')
        tab = tabulate(
            [[task, tab_iou[task][0], tab_iou[task][1]] +
             [f'{tab_iou[task][i] / tab_iou[task][0] * 100:.2f}' for i in range(2, len(tab_iou[task]))] +
             (['all', tab_iou_all[0], tab_iou_all[1]] + [f'{tab_iou_all[i] / tab_iou_all[0] * 100:.2f}' for i in range(2, len(tab_iou_all))])[0:0] 
             for task in tasks if task in tab_iou] +
            [['all', tab_iou_all[0], tab_iou_all[1]] +
             [f'{tab_iou_all[i] / tab_iou_all[0] * 100:.2f}' for i in range(2, len(tab_iou_all))]],
            headers=['Task', '#Samples', 'Failed'] + [f'R{k}@{t}' for k in top_k for t in thres] + ['mIoU'],
            tablefmt='pretty',
            stralign='left')
        nncore.log(tab)
    else:
        nncore.log("\n[Note] No temporal grounding data found or parsed (Spatial BBox cannot be eval with Temporal IoU).")

    if tab_ans_all[0] > 0:
        nncore.log('\nQA:')
        tab = tabulate(
            [[task, tab_ans[task][0], tab_ans[task][1]] +
             [f'{tab_ans[task][i] / tab_ans[task][0] * 100:.2f}' for i in range(2, 5)]
             for task in tasks if task in tab_ans] +
            [['all', tab_ans_all[0], tab_ans_all[1]] +
             [f'{tab_ans_all[i] / tab_ans_all[0] * 100:.2f}' for i in range(2, 5)]],
            headers=['Task', '#Samples', 'Failed', 'Acc', 'Acc (IoU >= 0.5)', 'Acc (IoP >= 0.5)'],
            tablefmt='pretty',
            stralign='left')
        nncore.log(tab)
    else:
        nncore.log("\n[Warning] No QA data found. Check if 'response' or 'model_pred' exists in JSON.")