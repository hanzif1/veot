# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.
# Modified for Grid-VLM: A Training-Free Spatial-Temporal Interface.

import argparse
import copy
import json
import gc
import os
import traceback
import re
import math
import numpy as np
from contextlib import nullcontext

import nncore
import torch
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
    parser.add_argument('--max_retries', type=int, default=5, help='Max retries for OOM errors')
    
    # [Grid-VLM Specific Arguments]
    parser.add_argument('--grid_side', type=int, default=3, 
                        help='Grid side length K (Total frames = K*K). e.g., 3 for 3x3 grid.')
    parser.add_argument('--search_depth', type=int, default=1, 
                        help='Depth of iterative refinement. 1=Single Stage, 2=Zoom-in once.')
    
    args = parser.parse_args()
    return args

# ================= [核心函数：支持 Zoom-in 的空间网格生成] =================
def create_spatial_grid(video_path, num_grids=9, span=None):
    """
    读取视频生成拼图。
    - 如果 span 为 None，读取全片。
    - 如果 span 为 [start, end]，只读取该时间段内的帧（Zoom-in）。
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        video_total_duration = total_frames / fps
        
        # 1. 确定采样范围 (帧索引)
        if span is None:
            # 默认：全片采样
            start_sec, end_sec = 0.0, video_total_duration
            start_idx, end_idx = 0, total_frames - 1
        else:
            # Zoom-in：局部采样
            start_sec, end_sec = span
            # 边界保护
            start_sec = max(0.0, start_sec)
            end_sec = min(video_total_duration, end_sec)
            
            start_idx = int(start_sec * fps)
            end_idx = int(end_sec * fps)
            
            # 索引边界保护
            start_idx = min(max(0, start_idx), total_frames - 1)
            end_idx = min(max(start_idx + 1, end_idx), total_frames - 1)

        # 当前片段的实际时长
        current_duration = end_sec - start_sec
        if current_duration <= 0: return None, [], 0

        # 2. 均匀采样
        indices = np.linspace(start_idx, end_idx, num_grids).astype(int)
        frames = vr.get_batch(indices).asnumpy() # (N, H, W, C)
        
        # 3. 计算拼图尺寸 & 动态缩放 (防止 Zoom-in 后图片过大)
        grid_side = math.ceil(math.sqrt(num_grids))
        h, w, _ = frames[0].shape
        
        if grid_side > 4: # 如果格子很多(如16x16)，缩小单帧
            scale = 4.0 / grid_side 
            new_w, new_h = int(w * scale), int(h * scale)
            # 保证最小尺寸
            new_w, new_h = max(32, new_w), max(32, new_h)
            resized_frames = []
            for f in frames:
                img = Image.fromarray(f).resize((new_w, new_h))
                resized_frames.append(np.array(img))
            frames = np.array(resized_frames)
            h, w = new_h, new_w

        # 4. 拼图
        grid_img = Image.new('RGB', (w * grid_side, h * grid_side))
        time_ranges = [] 
        
        # 每个格子代表的时长
        segment_len = current_duration / num_grids

        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            row = i // grid_side
            col = i % grid_side
            grid_img.paste(img, (col * w, row * h))
            
            # 5. 计算绝对时间范围 (关键步骤)
            # 第 i 个格子的中心点（相对于当前片段 start_sec）
            local_center = (i + 0.5) * segment_len
            abs_center = start_sec + local_center
            
            # 扩展一下范围作为预测结果
            t_s = max(0, abs_center - segment_len / 2)
            t_e = min(video_total_duration, abs_center + segment_len / 2)
            time_ranges.append([t_s, t_e])
            
        return grid_img, time_ranges, current_duration

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None, [], 0
# =========================================================================

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
    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device

    # 加载其他角色的 LoRA (可选)
    if args.model_pla_path is not None:
        adapter_path = nncore.join(args.model_pla_path, 'planner')
        if nncore.is_dir(adapter_path):
            model.load_adapter(adapter_path, adapter_name='planner')
            adapter_state['planner'] = True

    if args.model_ver_path is not None:
        adapter_path = nncore.join(args.model_ver_path, 'verifier')
        if nncore.is_dir(adapter_path):
            model.load_adapter(adapter_path, adapter_name='verifier')
            adapter_state['verifier'] = True

    if args.model_ans_path is not None:
        adapter_path = nncore.join(args.model_ans_path, 'answerer')
        if nncore.is_dir(adapter_path):
            model.load_adapter(adapter_path, adapter_name='answerer')
            adapter_state['answerer'] = True

    dataset_cls = DATASETS.get(args.dataset)

    print(f'Loading Dataset: {args.dataset}({args.split})')
    annos = dataset_cls.load_annos(split=args.split)
    if args.chunk > 1:
        annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]
    print(f'Total samples: {len(annos)}')

    # ================= [断点续传] =================
    dumps = []
    processed_count = 0
    if nncore.is_file(pred_path):
        try:
            dumps = nncore.load(pred_path)
            processed_count = len(dumps)
            print(f"Resuming from index {processed_count}...")
        except Exception:
            dumps = []
            processed_count = 0
    # ============================================

    DEFAULT_MAX_FRAMES = 64
    DEFAULT_MAX_PIXELS = 64 * 28 * 28 

    prog_bar = nncore.ProgressBar(range(len(annos)))
    for i in prog_bar:
        
        if i < processed_count:
            prog_bar.update()
            continue

        # 重置配置
        trial_max_frames = DEFAULT_MAX_FRAMES
        trial_max_pixels = DEFAULT_MAX_PIXELS

        # ================= [OOM 重试循环] =================
        retry_cnt = 0
        success = False
        dump = None 

        while retry_cnt <= args.max_retries:
            try:
                dump = copy.deepcopy(annos[i])
                video_path = dump['video_path']

                duration = dump.get('duration')
                if duration is None:
                    duration = get_duration(video_path, num_threads=args.num_threads)
                    dump['duration'] = duration

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
                    
                    messages = [{
                        'role': 'user',
                        'content': [{
                            'type': 'video', 'video': video_path, 'num_threads': args.num_threads,
                            'min_pixels': 36 * 28 * 28, 'max_pixels': trial_max_pixels,
                            'max_frames': trial_max_frames, 'fps': 1.0
                        }, {
                            'type': 'text', 'text': PLANNER_PROMPT.format(question)
                        }]
                    }]

                    text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    images, videos = process_vision_info(messages)
                    data = processor(text=[text], images=images, videos=videos, return_tensors='pt').to(device)

                    model.base_model.disable_adapter_layers()
                    model.base_model.enable_adapter_layers()
                    model.set_adapter('planner')

                    output_ids = model.generate(**data, do_sample=False, max_new_tokens=256)
                    response = processor.decode(output_ids[0, data.input_ids.size(1):], clean_up_tokenization_spaces=False)
                    dump['planner_response'] = response

                    try:
                        parsed = json.loads(response)
                        action = parsed[0] if isinstance(parsed, list) else parsed
                        if args.auto_rephrasing and action['type'].lower() == 'grounder' and action['value']:
                            query = action['value']
                        elif args.auto_planning and action['type'].lower() == 'answerer':
                            do_grounding = False
                    except Exception:
                        pass
                    
                    # 资源清理
                    if 'data' in locals(): del data
                    if 'images' in locals(): del images
                    if 'videos' in locals(): del videos
                    torch.cuda.empty_cache()

                # =================== Agent 2: Grounder (Grid-VLM Core) ===================
                if do_grounding:
                    dump['agents'].append('grounder')
                    query = parse_query(query)
                    
                    K = args.grid_side
                    num_grids = K * K
                    max_depth = args.search_depth  # 获取深度参数
                    
                    print(f"[{i}] Running Grounder | Grid: {K}x{K} | Depth: {max_depth}...")

                    # 初始化：搜索范围从全片开始
                    current_span = [0, duration]
                    final_pred = [[0, duration]]
                    final_conf = [0.0]
                    last_response = ""

                    # --- [迭代循环：核心逻辑] ---
                    for depth in range(max_depth):
                        # print(f"    [Level {depth}] Searching in span: {current_span}")
                        
                        # 调用 create_spatial_grid，传入当前的 span
                        grid_image, time_ranges, seg_duration = create_spatial_grid(
                            video_path, num_grids=num_grids, span=current_span
                        )
                        
                        if grid_image is None:
                            print("    Error creating grid, stopping refinement.")
                            break

                        # Prompt
                        spatial_prompt = (
                            f"The image shows {num_grids} frames extracted from a video segment, arranged in a {K}x{K} grid.\n"
                            f"The indices are ordered from top-left (0) to bottom-right ({num_grids-1}).\n"
                            f"Question: Which grid cell best matches the query '{query}'?\n"
                            f"Respond with the index number (0-{num_grids-1}) of the best matching cell."
                        )

                        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': grid_image}, {'type': 'text', 'text': spatial_prompt}]}]
                        
                        text = processor.apply_chat_template(messages, add_generation_prompt=True)
                        images, videos = process_vision_info(messages)
                        # 这里传入 image
                        inputs = processor(text=[text], images=images, videos=videos, return_tensors='pt').to(device)
                        
                        with model.disable_adapter(): 
                            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=10)
                        
                        response = processor.decode(output_ids[0, inputs.input_ids.size(1):], skip_special_tokens=True).strip()
                        last_response = response
                        # print(f"    VLM Response: {response}")
                        
                        # 解析结果
                        match = re.search(r'\d+', response)
                        if match:
                            idx = int(match.group())
                            if 0 <= idx < len(time_ranges):
                                selected_span = time_ranges[idx]
                                # print(f"    Selected Index {idx} -> New Span: {selected_span}")
                                
                                # [核心] 更新搜索范围，准备下一次循环（Zoom-in）
                                current_span = selected_span
                                final_pred = [selected_span]
                                final_conf = [1.0]
                            else:
                                # print("    Index out of bounds.")
                                break 
                        else:
                            # print("    Failed to parse index.")
                            break 
                        
                        # 清理 Grounder 临时显存
                        vars_to_del = ['inputs', 'images', 'videos', 'output_ids', 'grid_image']
                        for var_name in vars_to_del:
                            if var_name in locals(): del locals()[var_name]
                        torch.cuda.empty_cache()

                    dump['grounder_response'] = last_response
                    dump['grounder_success'] = True 
                    dump['pred'] = final_pred
                    dump['conf'] = final_conf

                # =================== Agent 3: Verifier ===================
                if do_grounding and adapter_state['verifier'] and len(final_pred) > 0:
                    dump['agents'].append('verifier')
                    # ... (Verifier 逻辑简化版，复用 VideoMind 原有逻辑) ...
                    # 为简单起见，这里省略复杂的 Token 插入，假设仅用于占位或简单评分
                    # 如果需要精确的 Verifier，请确保你的 VideoMind 依赖完整
                    dump['probs'] = [1.0] * len(final_pred)

                # =================== Agent 4: Answerer ===================
                if do_answering:
                    # print(f"[{i}] Running Answerer...")
                    dump['agents'].append('answerer')
                    
                    selected = final_pred[0] if 'pred' in dump and len(dump['pred']) > 0 else [0, duration]
                    s, e = selected[0], selected[1]

                    min_len = getattr(dataset_cls, 'MIN_LEN', 5.0)
                    if e - s < min_len:
                        center = (s + e) / 2
                        s = max(0, center - min_len / 2)
                        e = min(duration, center + min_len / 2)

                    messages = [{
                        'role': 'user',
                        'content': [{
                            'type': 'video', 'video': video_path, 'num_threads': args.num_threads,
                            'video_start': s, 'video_end': e,
                            'min_pixels': getattr(dataset_cls, 'MIN_PIXELS', 128) * 28 * 28,
                            'max_pixels': trial_max_pixels, 'max_frames': trial_max_frames,
                            'fps': getattr(dataset_cls, 'FPS', 2.0)
                        }, {
                            'type': 'text', 'text': prompt
                        }]
                    }]

                    text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    text += 'Best Option: (' if args.style == 'mcq' else ''
                    images, videos = process_vision_info(messages)
                    data = processor(text=[text], images=images, videos=videos, return_tensors='pt').to(device)

                    if adapter_state['answerer']:
                        model.base_model.disable_adapter_layers()
                        model.base_model.enable_adapter_layers()
                        model.set_adapter('answerer')
                        context = nullcontext
                    else:
                        context = model.disable_adapter

                    with context():
                        output_ids = model.generate(**data, do_sample=False, max_new_tokens=256)

                    response = processor.decode(output_ids[0, data.input_ids.size(1):], clean_up_tokenization_spaces=False)
                    # print(f"  Answer: {response}")
                    dump['answerer_response'] = response
                    dump['response'] = response
                    
                    if 'data' in locals(): del data
                    if 'images' in locals(): del images
                    if 'videos' in locals(): del videos
                    torch.cuda.empty_cache()

                success = True
                break # 跳出重试循环
            
            except torch.OutOfMemoryError:
                retry_cnt += 1
                print(f"\n[Warning] CUDA OOM. Retry {retry_cnt}/{args.max_retries}")
                trial_max_frames = int(trial_max_frames * 0.7)
                trial_max_pixels = int(trial_max_pixels * 0.8)
                
                if trial_max_frames < 8:
                    if dump is None: dump = copy.deepcopy(annos[i])
                    dump['error'] = 'OOM_Fail'
                    success = True
                    break

                # 显式清理所有可能的变量
                vars_to_del = ['data', 'images', 'videos', 'output_ids', 'grid_image', 'inputs']
                for var_name in vars_to_del:
                    if var_name in locals(): del locals()[var_name]
                gc.collect()
                torch.cuda.empty_cache()
            
            except Exception as e:
                print(f"\n[Error] {e}")
                traceback.print_exc()
                if dump is None: dump = copy.deepcopy(annos[i])
                dump['error'] = str(e)
                success = True
                break

        if success and dump is not None:
            dumps.append(dump)
            nncore.dump(dumps, pred_path)
            processed_count += 1
        
        gc.collect()
        torch.cuda.empty_cache()
        prog_bar.update()

    print(f"\nInference complete. Results saved to {pred_path}")