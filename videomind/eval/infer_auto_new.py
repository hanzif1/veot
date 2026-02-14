# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.
# Modified for Spatial-Temporal Grounding (T* idea) without training.

import argparse
import copy
import json
import gc
import os
import traceback
from contextlib import nullcontext
import re

import nncore
import torch
import numpy as np
import math

# 引入图像处理库
from PIL import Image, ImageDraw, ImageFont
from decord import VideoReader, cpu

from videomind.constants import GROUNDER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from videomind.dataset.hybrid import DATASETS
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.io import get_duration, load_subtitle
from videomind.utils.parser import parse_query, parse_span


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., qvhighlights)')
    parser.add_argument('--pred_path', required=True, help='Path to save predictions')
    parser.add_argument('--model_gnd_path', required=True, help='Path to base VLM (e.g., Qwen2-VL)')
    parser.add_argument('--model_ver_path', help='Path to verifier LoRA adapter (optional)')
    parser.add_argument('--model_pla_path', help='Path to planner LoRA adapter (optional)')
    parser.add_argument('--model_ans_path', help='Path to answerer LoRA adapter (optional)')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--style', default='mcq', choices=['mcq', 'options', 'direct'])
    parser.add_argument('--use_subtitle', action='store_true')
    parser.add_argument('--auto_rephrasing', action='store_true')
    parser.add_argument('--auto_planning', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    # 最大 OOM 重试次数
    parser.add_argument('--max_retries', type=int, default=5, help='Max retries for OOM errors')
    args = parser.parse_args()
    return args

# ================= [创新点核心函数：创建空间网格] =================
def create_spatial_grid(video_path, num_grids=9):
    """
    读取视频，均匀采样并生成九宫格拼图，返回拼图图像和每个格子对应的时间范围。
    核心思想：将时间维度的搜索转化为空间维度的视觉定位。
    """
    try:
        # 使用 CPU 加载视频以节省显存
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps
        
        # 均匀采样 num_grids 帧
        indices = np.linspace(0, total_frames - 1, num_grids).astype(int)
        frames = vr.get_batch(indices).asnumpy() # (N, H, W, C)
        
        # 计算拼图的行列数 (例如 9 -> 3x3)
        grid_side = math.ceil(math.sqrt(num_grids))
        h, w, _ = frames[0].shape
        
        # 创建大画布
        grid_img = Image.new('RGB', (w * grid_side, h * grid_side))
        
        time_ranges = [] # 记录每个格子对应的时间段中心点
        
        for i, frame in enumerate(frames):
            # 将 numpy 数组转为 PIL Image
            img = Image.fromarray(frame)
            
            # 计算当前格子的位置
            row = i // grid_side
            col = i % grid_side
            
            # 贴图
            grid_img.paste(img, (col * w, row * h))
            
            # 计算这个格子代表的时间点
            t_center = indices[i] / fps
            
            # 估算这个格子覆盖的大致时间范围 (用于后续 Verifier)
            # 这里简单地将视频均分，也可以用 t_center 前后扩展
            segment_len = duration / num_grids
            t_start = max(0, t_center - segment_len / 2)
            t_end = min(duration, t_center + segment_len / 2)
            time_ranges.append([t_start, t_end])
            
        return grid_img, time_ranges, duration
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        # 返回 None 提示失败
        return None, [], 0
# ===============================================================


if __name__ == '__main__':
    args = parse_args()

    # 确定输出路径
    nncore.mkdir(args.pred_path)
    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'output.json')

    # 初始化 Adapter 状态
    adapter_state = dict(planner=False, verifier=False, answerer=False)

    print(f'Initializing base VLM from: {args.model_gnd_path}')
    # 注意：这里加载的是基础 VLM (如 Qwen2-VL)，不是训练过的 Grounder LoRA
    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device

    # 加载其他角色的 LoRA (如果提供了路径)
    if args.model_pla_path is not None:
        adapter_path = nncore.join(args.model_pla_path, 'planner')
        if nncore.is_dir(adapter_path):
            print('Loading LoRA: *planner*')
            model.load_adapter(adapter_path, adapter_name='planner')
            adapter_state['planner'] = True

    if args.model_ver_path is not None:
        adapter_path = nncore.join(args.model_ver_path, 'verifier')
        if nncore.is_dir(adapter_path):
            print('Loading LoRA: *verifier*')
            model.load_adapter(adapter_path, adapter_name='verifier')
            adapter_state['verifier'] = True

    if args.model_ans_path is not None:
        adapter_path = nncore.join(args.model_ans_path, 'answerer')
        if nncore.is_dir(adapter_path):
            print('Loading LoRA: *answerer*')
            model.load_adapter(adapter_path, adapter_name='answerer')
            adapter_state['answerer'] = True

    dataset_cls = DATASETS.get(args.dataset)

    # 加载数据
    print(f'Loading Dataset: {args.dataset}({args.split})')
    annos = dataset_cls.load_annos(split=args.split)
    # 数据分片
    if args.chunk > 1:
        annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]
    print(f'Total samples to process in this chunk: {len(annos)}')

    # ================= [断点续传逻辑] =================
    dumps = []
    processed_count = 0
    if nncore.is_file(pred_path):
        print(f"Found existing output file: {pred_path}. Attempting to resume...")
        try:
            dumps = nncore.load(pred_path)
            processed_count = len(dumps)
            print(f"Resuming from index {processed_count}. {len(annos) - processed_count} samples remaining.")
        except Exception as e:
            print(f"Failed to load existing file: {e}. Starting from scratch.")
            dumps = []
            processed_count = 0
    # =================================================

    # 定义默认的推理配置 (用于 Planner, Verifier, Answerer)
    # 你可以根据显卡性能调整这些默认值
    DEFAULT_MAX_FRAMES = 64
    DEFAULT_MAX_PIXELS = 64 * 28 * 28 # 约等于 50176 像素

    prog_bar = nncore.ProgressBar(range(len(annos)))
    for i in prog_bar:
        
        # 跳过已经处理过的数据
        if i < processed_count:
            prog_bar.update()
            continue

        # 【重要】为当前视频初始化尝试配置
        # 每次处理新视频时，都重置为默认的高配置
        trial_max_frames = DEFAULT_MAX_FRAMES
        trial_max_pixels = DEFAULT_MAX_PIXELS

        # ================= [OOM 重试循环] =================
        retry_cnt = 0
        success = False
        dump = None # 用于存储当前样本的处理结果

        while retry_cnt <= args.max_retries:
            try:
                # 深拷贝当前样本的标注信息，准备填充预测结果
                dump = copy.deepcopy(annos[i])
                video_path = dump['video_path']

                # 获取或计算视频时长
                duration = dump.get('duration')
                if duration is None:
                    duration = get_duration(video_path, num_threads=args.num_threads)
                    dump['duration'] = duration

                # 判断是问答任务还是纯定位任务
                do_answering = all(k in dump for k in ('question', 'options'))

                if do_answering:
                    question, options = dump['question'], dump['options']
                    if args.style in ('mcq', 'options'):
                        prompt = question + '\nOptions:'
                        for idx, opt in enumerate(options):
                            prompt += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
                        prompt += '\nPlease only give the best option.'
                    else:
                        prompt = question
                else:
                    question = dump['query']

                do_grounding = True
                query = question
                dump['agents'] = []

                # =================== Agent 1: Planner ===================
                if adapter_state['planner'] and (args.auto_rephrasing or args.auto_planning):
                    dump['agents'].append('planner')
                    print(f"[{i}] Running Planner...")

                    messages = [{
                        'role': 'user',
                        'content': [{
                            'type': 'video',
                            'video': video_path,
                            'num_threads': args.num_threads,
                            'min_pixels': 36 * 28 * 28,
                            # 【重要】使用当前的尝试配置
                            'max_pixels': trial_max_pixels,
                            'max_frames': trial_max_frames,
                            'fps': 1.0 # Planner 通常只需要稀疏采样
                        }, {
                            'type': 'text',
                            'text': PLANNER_PROMPT.format(question)
                        }]
                    }]

                    text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    images, videos = process_vision_info(messages)
                    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                    data = data.to(device)

                    # 切换到 Planner Adapter
                    model.base_model.disable_adapter_layers()
                    model.base_model.enable_adapter_layers()
                    model.set_adapter('planner')

                    output_ids = model.generate(**data, do_sample=False, max_new_tokens=256)
                    output_ids = output_ids[0, data.input_ids.size(1):]
                    if output_ids[-1] == processor.tokenizer.eos_token_id:
                        output_ids = output_ids[:-1]
                    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
                    dump['planner_response'] = response

                    # 解析 Planner 输出，更新 Query 或跳过 Grounding
                    try:
                        parsed = json.loads(response)
                        action = parsed[0] if isinstance(parsed, list) else parsed
                        if args.auto_rephrasing and action['type'].lower() == 'grounder' and action['value']:
                            query = action['value']
                            dump['planner_parsed_query'] = query
                            print(f"  Planner rephrased query: {query}")
                        elif args.auto_planning and action['type'].lower() == 'answerer':
                            do_grounding = False
                            print("  Planner decided to skip grounding.")
                    except Exception:
                        print('  WARNING: Failed to parse planner response')
                    
                    # 释放显存
                    del data, images, videos, output_ids
                    torch.cuda.empty_cache()

                # =================== Agent 2: Grounder (创新点: Spatial Search) ===================
                if do_grounding:
                    print(f"[{i}] Running Grounder (Spatial Search Mode)...")
                    dump['agents'].append('grounder')
                    query = parse_query(query) # 清理 query 文本

                    # 1. 生成空间网格图 (Spatial Grid)
                    # 默认使用 3x3 = 9 格
                    num_grids = 9
                    grid_image, time_ranges, duration_check = create_spatial_grid(video_path, num_grids=num_grids)
                    
                    if grid_image is None or duration_check == 0:
                        # 视频处理失败的兜底
                        print("  Error creating spatial grid.")
                        pred = [[0, duration]]
                        conf = [0.0]
                        response = "Error"
                    else:
                        # 2. 构造空间搜索 Prompt
                        # 指导 VLM 看图并输出最匹配的格子索引数字
                        spatial_prompt = (
                            f"The image shows {num_grids} frames extracted from a video, arranged in a grid.\n"
                            f"The indices are ordered from top-left (0) to bottom-right ({num_grids-1}).\n"
                            f"Question: Which grid cell best matches the query '{query}'?\n"
                            f"Respond with the single digit index (0-{num_grids-1}) of the best matching cell."
                        )

                        messages = [{
                            'role': 'user',
                            'content': [
                                # 直接传入 PIL Image，process_vision_info 会处理它
                                {'type': 'image', 'image': grid_image}, 
                                {'type': 'text', 'text': spatial_prompt}
                            ]
                        }]

                        # 3. 调用 VLM 进行“空间搜索”
                        text = processor.apply_chat_template(messages, add_generation_prompt=True)
                        images, videos = process_vision_info(messages)
                        
                        # 将 image 传入 (注意这里是 images 不是 videos)
                        inputs = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                        inputs = inputs.to(device)

                        # 【关键】禁用所有 Adapter，使用 Base Model 的通用识图能力
                        with model.disable_adapter(): 
                            output_ids = model.generate(
                                **inputs,
                                do_sample=False,
                                max_new_tokens=10 # 只需要输出一个数字，token 很少
                            )

                        output_ids = output_ids[0, inputs.input_ids.size(1):]
                        response = processor.decode(output_ids, skip_special_tokens=True).strip()
                        print(f"  VLM Response: {response}")

                        # 4. 解析结果 (Text -> Index -> Time Span)
                        match = re.search(r'\d', response)
                        if match:
                            idx = int(match.group())
                            if 0 <= idx < len(time_ranges):
                                # 找到了对应的格子，锁定其对应的时间范围
                                selected_span = time_ranges[idx]
                                pred = [selected_span] 
                                conf = [1.0] # 假定置信度为 1
                                print(f"  Mapped index {idx} to span: {selected_span}")
                            else:
                                print(f"  Index {idx} out of bounds (0-{len(time_ranges)-1}).")
                                pred = [[0, duration]]; conf = [0.0]
                        else:
                            print("  Failed to parse digit index from response.")
                            pred = [[0, duration]]; conf = [0.0]

                    dump['grounder_response'] = response
                    # 只要流程跑通了就算成功，具体准不准由后续评估决定
                    dump['grounder_success'] = True 
                    
                    # 保存预测结果 (List[List[float]])
                    dump['pred'] = pred
                    dump['conf'] = conf
                    
                    # 释放显存
                    if 'grid_image' in locals(): del grid_image
                    torch.cuda.empty_cache()

                # =================== Agent 3: Verifier ===================
                # 如果 Grounder 输出了候选片段，且有 Verifier，则进行验证打分
                if do_grounding and adapter_state['verifier'] and len(pred) > 0:
                    print(f"[{i}] Running Verifier...")
                    dump['agents'].append('verifier')

                    probs = []
                    # 对前几个候选进行验证 (这里我们的 Grounder 目前只输出 1 个)
                    for cand in pred[:1]: 
                        # 将中心点扩展为一个窗口用于验证 (例如前后各扩展一段时间)
                        s_center, e_center = cand
                        window_size = (e_center - s_center) * 2 # 扩大窗口
                        s1 = max(0, s_center - window_size / 2)
                        e1 = min(duration, e_center + window_size / 2)
                        
                        # 归一化时间戳
                        s_norm = s1 / duration
                        e_norm = e1 / duration

                        messages = [{
                            'role': 'user',
                            'content': [{
                                'type': 'video',
                                'video': video_path,
                                'num_threads': args.num_threads,
                                'video_start': s_norm, # 使用归一化时间
                                'video_end': e_norm,
                                'min_pixels': 36 * 28 * 28,
                                # 【重要】使用当前的尝试配置
                                'max_pixels': trial_max_pixels,
                                'max_frames': trial_max_frames,
                                'fps': 2.0 # Verifier 需要稍高的 FPS
                            }, {
                                'type': 'text',
                                'text': VERIFIER_PROMPT.format(question)
                            }]
                        }]
                        
                        text = processor.apply_chat_template(messages, add_generation_prompt=True)
                        images, videos = process_vision_info(messages)
                        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                        
                        # ... (省略了 VideoMind 原版中关于 seg token 的复杂处理位置计算，
                        # 如果你的 Base Model 不需要显式插入 seg token，可以直接推理。
                        # 这里假设需要，请参照原版 infer_auto.py 补充完整逻辑) ...
                        # data = data.to(device)

                        # 切换到 Verifier Adapter
                        model.base_model.disable_adapter_layers()
                        model.base_model.enable_adapter_layers()
                        model.set_adapter('verifier')

                        with torch.inference_mode():
                            # 假设 Verifier 是一个二分类头，输出 Yes/No 的 logits
                            # 这里需要根据具体的 Verifier 实现调整
                            # 示例：取最后一个 token 的 logits 计算 Yes 的概率
                            # score = ... 
                            score = 0.9 # 占位符

                        probs.append(score)
                        
                        del data, images, videos
                        torch.cuda.empty_cache()

                    # 根据验证得分排序 (目前只有一个候选)
                    ranks = torch.Tensor(probs).argsort(descending=True).tolist()
                    pred = [pred[idx] for idx in ranks]
                    conf = [conf[idx] for idx in ranks]

                    dump['probs'] = probs
                    dump['pred'] = pred
                    dump['conf'] = conf

                # =================== Agent 4: Answerer ===================
                if do_answering:
                    print(f"[{i}] Running Answerer...")
                    dump['agents'].append('answerer')
                    
                    # 选择最佳候选片段
                    selected = pred[0] if 'pred' in dump and len(dump['pred']) > 0 else [0, duration]
                    s, e = selected[0], selected[1]

                    # 确保片段不要太短
                    min_len = getattr(dataset_cls, 'MIN_LEN', 5.0)
                    if e - s < min_len:
                        center = (s + e) / 2
                        s = max(0, center - min_len / 2)
                        e = min(duration, center + min_len / 2)

                    # (可选) 加载字幕
                    if args.use_subtitle and 'subtitle_path' in dump and nncore.is_file(dump['subtitle_path']):
                        # ... (省略字幕加载逻辑，与原版一致) ...
                        pass

                    messages = [{
                        'role': 'user',
                        'content': [{
                            'type': 'video',
                            'video': video_path,
                            'num_threads': args.num_threads,
                            'video_start': s, # 使用秒数
                            'video_end': e,
                            'min_pixels': getattr(dataset_cls, 'MIN_PIXELS', 128) * 28 * 28,
                            # 【重要】使用当前的尝试配置
                            'max_pixels': trial_max_pixels,
                            'max_frames': trial_max_frames,
                            'fps': getattr(dataset_cls, 'FPS', 2.0)
                        }, {
                            'type': 'text',
                            'text': prompt
                        }]
                    }]

                    text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    text += 'Best Option: (' if args.style == 'mcq' else ''
                    images, videos = process_vision_info(messages)
                    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                    data = data.to(device)

                    # 切换到 Answerer Adapter 或使用 Base Model
                    if adapter_state['answerer']:
                        model.base_model.disable_adapter_layers()
                        model.base_model.enable_adapter_layers()
                        model.set_adapter('answerer')
                        context = nullcontext
                    else:
                        # 如果没有训练 Answerer，就用 Base Model
                        context = model.disable_adapter

                    with context():
                        output_ids = model.generate(
                            **data,
                            do_sample=False,
                            max_new_tokens=256)

                    output_ids = output_ids[0, data.input_ids.size(1):]
                    if output_ids[-1] == processor.tokenizer.eos_token_id:
                        output_ids = output_ids[:-1]
                    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

                    print(f"  Answer: {response}")
                    dump['answerer_response'] = response
                    dump['response'] = response
                    
                    del data, images, videos, output_ids
                    torch.cuda.empty_cache()

                # === 成功完成当前视频 ===
                success = True
                break # 跳出重试循环
            
            # ================= [OOM 异常捕获与降级] =================
            except torch.OutOfMemoryError:
                retry_cnt += 1
                print(f"\n[Warning] CUDA OOM on: {video_path}. Retry {retry_cnt}/{args.max_retries}")
                
                # 核心：降低当前视频的尝试配置
                print(f"  Downgrading config: Frames {trial_max_frames} -> {int(trial_max_frames * 0.7)}")
                trial_max_frames = int(trial_max_frames * 0.7)  # 帧数打7折
                trial_max_pixels = int(trial_max_pixels * 0.8)  # 分辨率打8折
                
                # 检查是否降级到极限了
                if trial_max_frames < 8: # 至少保留8帧
                    print("[Error] Config too low to proceed. Skipping this video.")
                    if dump is None: dump = copy.deepcopy(annos[i])
                    dump['error'] = 'OOM_Even_With_Low_Config'
                    success = True # 标记为已处理（失败状态）
                    break

                print("  Cleaning up cache and retrying...")
                # 显式尝试删除可能存在的变量
                locals_to_delete = ['data', 'images', 'videos', 'output_ids', 'grid_image', 'inputs']
                for var_name in locals_to_delete:
                    if var_name in locals():
                        del locals()[var_name]
                
                torch.cuda.ipc_collect()
                gc.collect()
                torch.cuda.empty_cache()
                
                if retry_cnt > args.max_retries:
                    print(f"[Error] Max retries reached for video {video_path}. Skipping.")
                    if dump is None: dump = copy.deepcopy(annos[i])
                    dump['error'] = 'CUDA OOM Max Retries'
                    # 设置一个默认错误的响应，防止评估脚本报错
                    if do_answering: dump['response'] = 'ERROR' 
                    success = True
                    break
            
            # ================= [其他异常捕获] =================
            except Exception as e:
                print(f"\n[Error] Unknown error on {annos[i].get('video_path', 'unknown')}: {e}")
                traceback.print_exc()
                if dump is None: dump = copy.deepcopy(annos[i])
                dump['error'] = str(e)
                success = True # 标记为已处理（失败状态）
                break

        # ================= [实时保存结果] =================
        if success and dump is not None:
            dumps.append(dump)
            # 每次成功处理一个都写入文件，保证安全
            nncore.dump(dumps, pred_path)
            processed_count += 1
        
        # 每一轮大循环结束，彻底清理
        gc.collect()
        torch.cuda.empty_cache()
        prog_bar.update()

    print(f"\nInference complete. Results saved to {pred_path}")