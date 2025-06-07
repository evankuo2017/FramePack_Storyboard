import http.server
import socketserver
import json
import os
import base64
import cgi
import urllib.parse
from datetime import datetime
import logging
import webbrowser
import threading
import time
import mimetypes
import subprocess
import queue
import sys
import numpy as np
from PIL import Image
import torch
import traceback

# 導入framepack_start_end.py中的功能
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from framepack_start_end import process_video as framepack_process_video
    print("成功導入framepack_start_end模組")
except ImportError as e:
    print(f"無法導入framepack_start_end模組: {e}")
    framepack_process_video = None

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 確保輸出目錄存在
OUTPUT_DIR = "storyboard_outputs"
JSON_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "json_output")
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "image_output")
VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "video_output")

for dir_path in [OUTPUT_DIR, JSON_OUTPUT_DIR, IMAGE_OUTPUT_DIR, VIDEO_OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"創建目錄: {dir_path}")

# 任務隊列和處理狀態
task_queue = queue.Queue()
task_status = {}  # 存儲任務狀態的字典
stop_processing = False

# 配置參數
default_params = {
    "prompt": "Character movements based on storyboard sequence",
    "n_prompt": "",
    "seed": 31337,
    "total_second_length": 5,
    "latent_window_size": 9,
    "steps": 25,
    "cfg": 1.0,
    "gs": 10.0,
    "rs": 0.0,
    "gpu_memory_preservation": 6,
    "use_teacache": True,
    "mp4_crf": 16
}

# 模擬 FramePack 處理函數
def process_video_task(task_id, start_image_path, end_image_path=None, params=None):
    """
    處理視頻生成任務
    
    Args:
        task_id: 任務ID
        start_image_path: 起始幀圖片路徑
        end_image_path: 結束幀圖片路徑 (可選)
        params: 其他參數字典
    """
    global task_status
    
    try:
        task_status[task_id]["status"] = "processing"
        task_status[task_id]["progress"] = 0
        task_status[task_id]["message"] = "啟動處理任務..."
        
        # 合併默認參數和提供的參數
        if params is None:
            params = {}
        actual_params = default_params.copy()
        actual_params.update(params)
        
        logger.info(f"開始處理任務 {task_id}，起始幀：{start_image_path}, 結束幀：{end_image_path}")
        
        # 檢查是否應該停止處理
        if stop_processing:
            task_status[task_id]["status"] = "cancelled"
            task_status[task_id]["message"] = "任務被取消"
            return None
        
        # 準備輸出文件名
        output_filename = os.path.join(VIDEO_OUTPUT_DIR, f"{task_id}.mp4")
        
        # 使用framepack_start_end.py中的功能
        if framepack_process_video:
            # 使用自定義進度回調來更新任務狀態
            def progress_callback(percentage, message):
                task_status[task_id]["progress"] = percentage
                task_status[task_id]["message"] = message
                logger.info(f"任務 {task_id} 進度: {percentage}% - {message}")
            
            # 調用framepack_start_end.py中的處理函數
            logger.info(f"使用framepack_process_video處理任務 {task_id}")
            result = framepack_process_video(
                start_image_path, 
                end_image_path, 
                progress_callback=progress_callback,
                prompt=actual_params["prompt"],
                n_prompt=actual_params["n_prompt"],
                seed=actual_params["seed"],
                total_second_length=actual_params["total_second_length"],
                latent_window_size=actual_params["latent_window_size"],
                steps=actual_params["steps"],
                cfg=actual_params["cfg"],
                gs=actual_params["gs"],
                rs=actual_params["rs"],
                gpu_memory_preservation=actual_params["gpu_memory_preservation"],
                use_teacache=actual_params["use_teacache"],
                mp4_crf=actual_params["mp4_crf"],
                output=output_filename
            )
            
            if result:
                task_status[task_id]["status"] = "completed"
                task_status[task_id]["progress"] = 100
                task_status[task_id]["message"] = "處理完成！"
                task_status[task_id]["output_file"] = result
                logger.info(f"任務 {task_id} 處理完成，輸出文件: {result}")
                return result
            else:
                task_status[task_id]["status"] = "error"
                task_status[task_id]["message"] = "處理失敗"
                logger.error(f"任務 {task_id} 處理失敗")
                return None
        else:
            # 如果無法導入framepack_process_video，則標記任務為錯誤
            logger.error(f"任務 {task_id} 無法處理: framepack_process_video 模組未成功導入。")
            task_status[task_id]["status"] = "error"
            task_status[task_id]["message"] = "處理模組不可用 (framepack_process_video missing)"
            task_status[task_id]["progress"] = 0 # 確保進度為0
            return None
        
    except Exception as e:
        logger.error(f"處理任務 {task_id} 出錯: {e}")
        traceback.print_exc()
        task_status[task_id]["status"] = "error"
        task_status[task_id]["message"] = f"錯誤: {str(e)}"
        return None

# 任務處理線程
def task_processor():
    global task_status, stop_processing
    
    logger.info("任務處理線程已啟動")
    
    while not stop_processing:
        try:
            # 從隊列獲取任務，非阻塞
            try:
                task = task_queue.get(block=False)
                task_id = task.get("id")
                logger.info(f"開始處理任務: {task_id}")
                
                # 處理任務
                start_image = task.get("start_image")
                end_image = task.get("end_image")
                params = task.get("params", {})
                
                process_video_task(task_id, start_image, end_image, params)
                
                # 標記任務完成
                task_queue.task_done()
                
            except queue.Empty:
                # 隊列為空，等待
                time.sleep(1)
                continue
                
        except Exception as e:
            logger.error(f"任務處理線程發生錯誤: {e}")
            traceback.print_exc()
            time.sleep(5)  # 出錯後稍微等待
    
    logger.info("任務處理線程已停止")

# 從故事板JSON生成任務
def create_tasks_from_storyboard(storyboard_file):
    """
    從故事板JSON文件生成視頻處理任務
    
    Args:
        storyboard_file: JSON文件路徑
    
    Returns:
        添加的任務ID列表，或錯誤信息
    """
    try:
        with open(storyboard_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        job_id = data.get("job_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        nodes = data.get("nodes", [])
        transitions = data.get("transitions", [])
        
        if len(nodes) < 2:
            error_msg = f"故事板 {storyboard_file} 節點少於2個，無法創建任務"
            logger.warning(error_msg)
            return {"success": False, "message": "Storyboard needs at least 2 nodes to generate videos."}
        
        # 檢查所有節點是否都有圖片
        missing_images = []
        for i, node in enumerate(nodes):
            has_image = node.get("hasImage", False)
            image_path = node.get("imagePath")
            
            if not has_image or not image_path:
                missing_images.append(i)
                continue
                
            # 檢查圖片文件是否存在
            full_path = os.path.join(IMAGE_OUTPUT_DIR, image_path)
            if not os.path.exists(full_path):
                alt_path = os.path.join(IMAGE_OUTPUT_DIR, os.path.basename(image_path))
                if not os.path.exists(alt_path):
                    missing_images.append(i)
        
        # 如果有節點缺少圖片，返回錯誤
        if missing_images:
            nodes_str = ", ".join([str(i) for i in missing_images])
            error_msg = f"節點 {nodes_str} 缺少圖片，請為所有節點添加圖片後再處理"
            logger.warning(error_msg)
            return {"success": False, "message": f"Nodes {nodes_str} are missing images. Please add images to all nodes before processing."}
        
        # 獲取節點圖像路徑
        task_ids = []
        
        # 為每一對連續節點創建任務
        for i in range(len(nodes) - 1):
            start_node = nodes[i]
            end_node = nodes[i + 1]
            
            # 獲取圖像路徑
            start_image_path = start_node.get("imagePath")
            
            # 處理圖片路徑
            if start_image_path:
                # 從文件名構建完整路徑
                start_image_path = os.path.join(IMAGE_OUTPUT_DIR, os.path.basename(start_image_path))
                
                if not os.path.exists(start_image_path):
                    logger.error(f"無法找到節點 {i} 的圖片: {start_image_path}")
                    continue
            
            # 處理結束節點圖像
            end_image_path = end_node.get("imagePath")
            
            if end_image_path:
                end_image_path = os.path.join(IMAGE_OUTPUT_DIR, os.path.basename(end_image_path))
                
                if not os.path.exists(end_image_path):
                    logger.warning(f"無法找到節點 {i+1} 的圖片: {end_image_path}")
                    end_image_path = None
            
            # 查找對應的轉場描述
            transition_text = ""
            for transition in transitions:
                if transition.get("from_node") == i and transition.get("to_node") == i + 1:
                    transition_text = transition.get("text", "")
                    break
            
            # 設置任務參數
            task_id = f"{job_id}_transition_{i}_to_{i+1}"
            
            # 設置任務時長（基於節點時間差）
            time_range = None
            for transition in transitions:
                if transition.get("from_node") == i and transition.get("to_node") == i + 1:
                    time_range = transition.get("time_range")
                    break
            
            second_length = 5  # 默認5秒
            if time_range and len(time_range) >= 2:
                second_length = max(1, time_range[1] - time_range[0])
            
            # 創建任務
            task = {
                "id": task_id,
                "start_image": start_image_path,
                "end_image": end_image_path,
                "params": {
                    "prompt": f"Character movement: {transition_text}" if transition_text else default_params["prompt"],
                    "total_second_length": second_length
                }
            }
            
            # 初始化任務狀態
            task_status[task_id] = {
                "status": "queued",
                "progress": 0,
                "message": "In queue",
                "created_at": datetime.now().isoformat()
            }
            
            # 添加到任務隊列
            task_queue.put(task)
            task_ids.append(task_id)
            
            logger.info(f"創建任務 {task_id}，從節點 {i} 到節點 {i+1}")
        
        return {"success": True, "task_ids": task_ids}
        
    except Exception as e:
        logger.error(f"從故事板創建任務時出錯: {e}")
        traceback.print_exc()
        return {"success": False, "message": f"Error creating tasks: {str(e)}"}

# 自定義 HTTP 請求處理器
class StoryboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # 處理根路徑請求，提供 storyboard.html
        if path == '/' or path == '':
            self.path = '/storyboard.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
            
        # 健康檢查
        elif path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = json.dumps({
                'status': 'ok',
                'message': '伺服器運行中'
            })
            self.wfile.write(response.encode('utf-8'))
            return
            
        # 獲取任務狀態
        elif path.startswith('/task_status'):
            query = urllib.parse.parse_qs(parsed_path.query)
            task_id = query.get('id', [''])[0]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            if task_id and task_id in task_status:
                response = json.dumps({
                    'status': 'success',
                    'task': task_status[task_id]
                })
            elif not task_id:
                # 返回所有任務狀態
                response = json.dumps({
                    'status': 'success',
                    'tasks': task_status
                })
            else:
                response = json.dumps({
                    'status': 'error',
                    'message': f'找不到任務 {task_id}'
                })
            
            self.wfile.write(response.encode('utf-8'))
            return
            
        # 列出所有故事板JSON文件
        elif path == '/list_storyboards':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                files = []
                for filename in os.listdir(JSON_OUTPUT_DIR):
                    if filename.endswith('.json'):
                        file_path = os.path.join(JSON_OUTPUT_DIR, filename)
                        file_stats = os.stat(file_path)
                        files.append({
                            'filename': filename,
                            'path': file_path,
                            'size': file_stats.st_size,
                            'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                        })
                
                response = json.dumps({
                    'status': 'success',
                    'files': files
                })
                
            except Exception as e:
                response = json.dumps({
                    'status': 'error',
                    'message': str(e)
                })
                
            self.wfile.write(response.encode('utf-8'))
            return

        # 新增：處理讀取特定故事板JSON文件的請求
        elif path == '/load_storyboard':
            query_components = urllib.parse.parse_qs(parsed_path.query)
            filename = query_components.get('file', [None])[0]

            if not filename:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response_data = {"status": "error", "message": "Filename parameter is missing"}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                return

            # 安全性檢查，防止路徑遍歷
            if '..' in filename or filename.startswith('/') or filename.startswith('\\\\'):
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response_data = {"status": "error", "message": "Invalid filename"}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                return

            file_path = os.path.join(JSON_OUTPUT_DIR, filename)

            if not os.path.exists(file_path):
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                # logger.info(f"Debug: Attempting to load file from: {os.path.abspath(file_path)}")
                response_data = {"status": "error", "message": f"Storyboard file '{filename}' not found at path: {file_path}"}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                return
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode('utf-8')) # 直接返回JSON檔案的內容
            
            except json.JSONDecodeError as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response_data = {"status": "error", "message": f"Error decoding JSON from file '{filename}': {str(e)}"}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response_data = {"status": "error", "message": f"Error loading storyboard file: {str(e)}"}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return
        
        # 視頻文件提供
        elif path.startswith('/video/'):
            video_name = path.replace('/video/', '')
            video_path = os.path.join(VIDEO_OUTPUT_DIR, video_name)
            
            if os.path.exists(video_path):
                self.send_response(200)
                content_type = mimetypes.guess_type(video_path)[0] or 'application/octet-stream'
                self.send_header('Content-type', content_type)
                self.send_header('Content-length', str(os.path.getsize(video_path)))
                self.end_headers()
                
                with open(video_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Video not found')
            return
            
        # 其他請求正常處理
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_OPTIONS(self):
        # 處理 CORS 預檢請求
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        # 處理儲存故事板的請求
        if self.path == '/save_storyboard':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                # 解析 JSON 資料
                data = json.loads(post_data.decode('utf-8'))
                logger.info(f"收到故事板資料: {len(data['nodes'])} 個節點")
                
                # 檢查是否至少有兩個節點
                if len(data['nodes']) < 2:
                    
                    self.send_response(400)  # Bad Request
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')  # CORS 支持
                    self.end_headers()
                    
                    response = json.dumps({
                        'status': 'error',
                        'message': "Storyboard needs at least 2 nodes to create transitions. Please add more nodes."
                    })
                    self.wfile.write(response.encode('utf-8'))
                    return
                
                # 檢查所有節點是否都有圖片
                missing_images = []
                for i, node in enumerate(data['nodes']):
                    if not node.get('hasImage', False) or ('imageData' not in node and 'imagePath' not in node):
                        missing_images.append(i)
                
                # 如果有節點缺少圖片，拒絕保存
                if missing_images:
                    nodes_str = ", ".join([str(i) for i in missing_images])
                    
                    self.send_response(400)  # Bad Request
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')  # CORS 支持
                    self.end_headers()
                    
                    response = json.dumps({
                        'status': 'error',
                        'message': f"Nodes {nodes_str} are missing images. Please add images to all nodes before saving."
                    })
                    self.wfile.write(response.encode('utf-8'))
                    return
                
                # 使用提供的檔名或生成新檔名
                file_name = data.get('file_name')
                if not file_name:
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_name = f"{len(data['nodes'])}_{now}.json"
                
                # 檔案完整路徑
                file_path = os.path.join(JSON_OUTPUT_DIR, file_name)
                
                # 處理圖片資料 (如果有的話)
                for node in data['nodes']:
                    if 'imageData' in node and 'imageType' in node:
                        # 儲存圖片到單獨的檔案
                        image_file_name = f"node_{node['index']}_{file_name.split('.')[0]}.png"
                        image_path = os.path.join(IMAGE_OUTPUT_DIR, image_file_name)
                        
                        # 解碼 base64 並儲存圖片
                        try:
                            with open(image_path, 'wb') as img_file:
                                img_file.write(base64.b64decode(node['imageData']))
                            
                            # 修改這裡: 直接使用圖片文件名作為路徑，避免多層嵌套路徑
                            # 在 JSON 中替換 base64 資料為檔案路徑
                            node['imagePath'] = image_file_name  # 只存儲文件名，不包含路徑
                            del node['imageData']
                            del node['imageType']
                            logger.info(f"儲存節點 {node['index']} 的圖片: {image_file_name}")
                        except Exception as e:
                            logger.error(f"儲存圖片時發生錯誤: {e}")
                
                # 儲存 JSON 檔案
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"故事板資料已儲存到: {file_path}")
                
                # 回傳成功訊息
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')  # CORS 支持
                self.end_headers()
                
                response = json.dumps({
                    'status': 'success',
                    'message': 'Storyboard data saved successfully',
                    'file_path': file_path
                })
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                logger.error(f"處理請求時發生錯誤: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = json.dumps({
                    'status': 'error',
                    'message': str(e)
                })
                self.wfile.write(response.encode('utf-8'))
        
        # 處理從故事板創建視頻任務的請求
        elif self.path == '/process_storyboard':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                storyboard_file = data.get('storyboard_file')
                
                if not storyboard_file:
                    raise ValueError("No storyboard file path provided")
                
                # 處理相對路徑
                if not os.path.isabs(storyboard_file):
                    storyboard_file = os.path.join(JSON_OUTPUT_DIR, storyboard_file)
                
                if not os.path.exists(storyboard_file):
                    raise FileNotFoundError(f"Storyboard file not found: {storyboard_file}")
                
                # 創建任務
                result = create_tasks_from_storyboard(storyboard_file)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')  # CORS 支持
                self.end_headers()
                
                if result.get("success", False):
                    response = json.dumps({
                        'status': 'success',
                        'message': f'Created {len(result["task_ids"])} video tasks',
                        'task_ids': result["task_ids"]
                    })
                else:
                    response = json.dumps({
                        'status': 'error',
                        'message': result.get("message", "Unknown error occurred")
                    })
                    
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                logger.error(f"處理故事板任務創建請求時發生錯誤: {e}")
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = json.dumps({
                    'status': 'error',
                    'message': str(e)
                })
                self.wfile.write(response.encode('utf-8'))
        
        else:
            # 其他 POST 請求返回 404
            self.send_response(404)
            self.end_headers()

# 自動開啟瀏覽器
def open_browser(PORT):
    """在啟動伺服器後自動開啟瀏覽器"""
    time.sleep(1)  # 等待伺服器啟動
    webbrowser.open(f'http://localhost:{PORT}')

if __name__ == '__main__':
    # 設置伺服器端口
    PORT = 7860
    
    # 啟動任務處理線程
    processor_thread = threading.Thread(target=task_processor, daemon=True)
    processor_thread.start()
    
    # 創建伺服器
    with socketserver.TCPServer(("", PORT), StoryboardHandler) as httpd:
        logger.info("啟動故事板系統...")
        logger.info(f"輸出目錄: {os.path.abspath(OUTPUT_DIR)}")
        logger.info(f"伺服器啟動在 http://localhost:{PORT}")
        
        # 在新執行緒中開啟瀏覽器，以免阻塞伺服器啟動
        threading.Thread(target=open_browser, args=(PORT,)).start()
        
        # 啟動伺服器
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("伺服器關閉")
            stop_processing = True  # 停止任務處理線程
            processor_thread.join(timeout=5)  # 等待任務處理線程結束
            httpd.server_close()