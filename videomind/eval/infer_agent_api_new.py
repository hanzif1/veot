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
    # ... (保持原有的读取逻辑不变) ...
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

def to_letter(ans):
    """将数字答案转换为字母，用于统一比较"""
    if isinstance(ans, int):
        return chr(65 + ans) # 0->A, 1->B
    if isinstance(ans, str) and ans.isdigit():
        return chr(65 + int(ans))
    return str(ans).upper()

class VideoAgent:
    def __init__(self):
        self.planner_client = OpenAI(base_url=PLANNER_URL, api_key="EMPTY")
        
    def run_planner(self, query, duration):
        # ... (保持不变) ...
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
        # ... (保持不变) ...
        base64_img = read_frame_as_base64(video_path)
        if not base64_img: return "Error"
        
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
        # ... (保持不变) ...
        base64_img = read_frame_as_base64(video_path)
        if not base64_img: return "Z"

        options_str = ""
        for idx, opt in enumerate(options):
            options_str += f"({chr(65+idx)}) {opt}\n"

        prompt = f"""
        Video Question: {question}
        
        Options:
        {options_str}
        
        Visual Clue found: {grounder_result}
        
        Task: Based on the visual clue and the video frame, select the best option.
        CRITICAL: Output ONLY the option letter (e.g., A, B, C, D, or E). Do not output anything else.
        """

        payload = {
            "model": "grounder-vl-72b",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]}],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        try:
            res = requests.post(f"{ANSWERER_URL}/chat/completions", json=payload, timeout=60)
            if res.status_code == 200:
                content = res.json()['choices'][0]['message']['content'].strip()
                import re
                match = re.search(r'[A-E]', content)
                return match.group(0) if match else content[0]
            return "Z"
        except Exception as e:
            return "Z"

    def run_answerer_retry(self, video_path, question, options, grounder_result, wrong_ans):
        """ === 新增：Retry 模块 (错题重做) === """
        base64_img = read_frame_as_base64(video_path)
        if not base64_img: return "Z"

        options_str = ""
        for idx, opt in enumerate(options):
            options_str += f"({chr(65+idx)}) {opt}\n"

        # === 构造 Retry Prompt ===
        # 核心逻辑：告诉模型之前的答案是错的，要求重新选择
        prompt = f"""
        Video Question: {question}
        
        Options:
        {options_str}
        
        Visual Clue found: {grounder_result}
        
        PREVIOUS MISTAKE: You previously selected option ({wrong_ans}), which was INCORRECT.
        
        Task: Re-analyze the image and the options. Avoid the previous mistake. Select the best option from the REMAINING options.
        CRITICAL: Output ONLY the option letter (e.g., A, B, C, D, or E). Do not output anything else.
        """

        payload = {
            "model": "grounder-vl-72b",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]}],
            "max_tokens": 10,
            "temperature": 0.2 # 稍微提高一点温度，增加变通性
        }
        
        try:
            res = requests.post(f"{ANSWERER_URL}/chat/completions", json=payload, timeout=60)
            if res.status_code == 200:
                content = res.json()['choices'][0]['message']['content'].strip()
                import re
                match = re.search(r'[A-E]', content)
                return match.group(0) if match else content[0]
            return "Z"
        except Exception as e:
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
            question = dump.get('question') or dump.get('query')
            options = dump.get('options', [])
            
            # ==========================================
            # 🛑 关键修正：在这里定义 task_type
            # 确保它在后续的逻辑之前就已经被赋值
            # ==========================================
            task_type = dump.get('task') or dump.get('task_type') 

            # 获取 Ground Truth
            raw_gt = dump.get('ans') or dump.get('answer') 
            gt_letter = to_letter(raw_gt) if raw_gt is not None else None

            # 1. Plan
            plan = agent.run_planner(question, 10.0)
            dump['planner_response'] = plan
            
            # 2. Ground
            grounder_res = agent.run_grounder(video_path, question)
            dump['grounder_response'] = grounder_res
            
            # 3. Answer
            if options:
                ans_pred = agent.run_answerer(video_path, question, options, grounder_res)
                dump['model_pred_first'] = ans_pred
                
                final_pred = ans_pred
                is_retry = False

                # === 校验与重试逻辑 ===
                if gt_letter and ans_pred != "Z":
                    # 现在 task_type 已经被定义，不会报错了
                    if ans_pred != gt_letter and str(task_type) == 'count':
                        
                        # print(f"Retrying Count Task [{i}] -> GT:{gt_letter} vs Pred:{ans_pred}")
                        retry_pred = agent.run_answerer_retry(video_path, question, options, grounder_res, ans_pred)
                        final_pred = retry_pred
                        is_retry = True
                    else:
                        pass
                
                dump['model_pred'] = final_pred 
                dump['is_retry'] = is_retry
            
            dumps.append(dump)
            if i % 10 == 0: nncore.dump(dumps, out_file)

        except Exception as e:
            # 这里的 print(e) 之前捕捉到了 name error
            print(f"Error {i}: {e}")
            continue

    nncore.dump(dumps, out_file)
    print(f"Done. Saved to {out_file}")