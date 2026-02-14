import argparse
import copy
import json
import os
import cv2
import numpy as np
import base64
import time
import requests
import nncore
from openai import OpenAI
from videomind.dataset.hybrid import DATASETS
from videomind.utils.io import get_duration

# === 配置 vLLM 端口 ===
PLANNER_URL = "http://localhost:8000/v1"
GROUNDER_URL = "http://localhost:8001/v1"
# Answerer 通常复用 Grounder 的视觉模型 (Qwen2-VL)，因为它需要看视频
ANSWERER_URL = "http://localhost:8001/v1" 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    return parser.parse_args()

def encode_image_base64(image):
    if image is None: return ""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def read_frame_as_base64(video_path):
    """读取中间帧转 Base64"""
    frame = None
    try:
        if isinstance(video_path, list):
            if len(video_path) == 0: return None
            mid_path = video_path[len(video_path) // 2]
            if os.path.exists(mid_path):
                frame = cv2.imread(mid_path)
        else:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ret, frame = cap.read()
                cap.release()
    except Exception as e:
        print(f"Read Frame Error: {e}")
        return None
    if frame is not None:
        return encode_image_base64(frame)
    return None

class VideoAgent:
    def __init__(self):
        self.planner_client = OpenAI(base_url=PLANNER_URL, api_key="EMPTY")
        
    def run_planner(self, query, duration):
        prompt = f"""Task: Plan steps to answer: "{query}". Video Duration: {duration}s. Output JSON list."""
        try:
            response = self.planner_client.chat.completions.create(
                model="planner-72b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=512
            )
            return response.choices[0].message.content
        except: return "[]"

    def run_grounder(self, video_path, query):
        base64_img = read_frame_as_base64(video_path)
        if not base64_img: return "Error"
        
        # 强制要求输出坐标
        prompt = f"Locate '{query}' in image. Output ONLY bbox [ymin, xmin, ymax, xmax]."
        
        payload = {
            "model": "grounder-vl-72b",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]}],
            "max_tokens": 128, "temperature": 0.1
        }
        try:
            res = requests.post(f"{GROUNDER_URL}/chat/completions", json=payload, timeout=120)
            res.raise_for_status()
            return res.json()['choices'][0]['message']['content']
        except: return "Error"

    def run_answerer(self, video_path, question, options, grounder_result):
        """ === 新增：Answerer (回答模块) === """
        base64_img = read_frame_as_base64(video_path)
        if not base64_img: return "Z" # 失败返回 Z

        # 1. 格式化选项 (A) Option1 (B) Option2 ...
        options_str = ""
        for idx, opt in enumerate(options):
            options_str += f"({chr(65+idx)}) {opt}\n"

        # 2. 构造提示词
        # 我们把 Grounder 找到的线索也告诉回答者，辅助它决策
        prompt = f"""
        Video Question: {question}
        
        Options:
        {options_str}
        
        Visual Clue found: {grounder_result}
        
        Task: Based on the visual clue and the video frame, select the best option.
        CRITICAL: Output ONLY the option letter (e.g., A, B, C, D, or E). Do not output anything else.
        """

        # 3. 发送请求
        payload = {
            "model": "grounder-vl-72b", # 复用视觉模型来做选择题
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]}],
            "max_tokens": 10, # 只需要输出一个字母，max_tokens设小点
            "temperature": 0.1
        }
        
        try:
            res = requests.post(f"{ANSWERER_URL}/chat/completions", json=payload, timeout=60)
            if res.status_code == 200:
                content = res.json()['choices'][0]['message']['content'].strip()
                # 简单清洗，只保留第一个字母
                import re
                match = re.search(r'[A-E]', content)
                return match.group(0) if match else content[0]
            return "Z"
        except Exception as e:
            # print(f"Answerer Error: {e}")
            return "Z"

if __name__ == '__main__':
    args = parse_args()
    if args.chunk > 1:
        out_file = nncore.join(args.pred_path, f'output_{args.index}.json')
    else:
        out_file = nncore.join(args.pred_path, 'output.json')

    dataset_cls = DATASETS.get(args.dataset)
    annos = dataset_cls.load_annos(split=args.split)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]
    
    agent = VideoAgent()
    dumps = []

    for i in nncore.ProgressBar(range(len(annos))):
        try:
            dump = copy.deepcopy(annos[i])
            video_path = dump['video_path']
            
            # 兼容 Query 和 Question 字段
            question = dump.get('question') or dump.get('query')
            options = dump.get('options', [])
            
            # 1. Plan
            plan = agent.run_planner(question, 10.0)
            dump['planner_response'] = plan
            
            # 2. Ground
            grounder_res = agent.run_grounder(video_path, question)
            dump['grounder_response'] = grounder_res
            
            # 3. Answer (新增步骤!)
            if options: # 如果有选项，就做选择题
                ans_pred = agent.run_answerer(video_path, question, options, grounder_res)
                dump['model_pred'] = ans_pred  # <--- 这就是你要的预测结果 (A/B/C/D)
                # print(f"GT: {dump.get('ans')} | Pred: {ans_pred}")
            
            dumps.append(dump)
            if i % 10 == 0: nncore.dump(dumps, out_file)

        except Exception as e:
            print(f"Error {i}: {e}")
            continue

    nncore.dump(dumps, out_file)
    print(f"Done. Saved to {out_file}")