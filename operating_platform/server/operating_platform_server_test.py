coding = "utf-8"

from gevent import monkey
monkey.patch_all()

from flask import Flask, jsonify, Response, request, session
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import os
import logging
import datetime
from flask_socketio import SocketIO, emit
import schedule
import requests
import json

import upload_to_nas


MACHINE_ID_FILE = './machine_id.txt'

class VideoStream:
    def __init__(self, stream_id, stream_name):
        self.stream_id = stream_id
        self.name = stream_name
        self.running = False
        self.frame_buffers = [None, None]  # 双缓冲
        self.buffer_index = 0
        self.lock = threading.Lock()
         
        

    def start(self):
        """启动视频流（仅标记为运行）"""
        if self.running:
            print(f"已经启动视频流")
            return True
        self.running = True
        return True
    
    def stop(self):
        """停止视频流"""
        self.running = False

    def update_frame(self, frame_data):
        """接收外部帧数据并更新当前帧"""
        if not self.running:
            return
        
        # 解码图像
        img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        
        # 压缩图像（可选）
        img = cv2.resize(img, (640, 480))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, jpeg = cv2.imencode('.jpg', img, encode_param)
        compressed_frame = jpeg.tobytes()

        with self.lock:
            self.buffer_index = 1 - self.buffer_index
            self.frame_buffers[self.buffer_index] = compressed_frame

    def get_frame(self):
        if not self.running:
            return self.generate_blank_frame()
        with self.lock:
            return self.frame_buffers[self.buffer_index]
    
    @staticmethod 
    def generate_blank_frame():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', blank)
        return jpeg.tobytes()

 
def get_machine_id():
    if os.path.exists(MACHINE_ID_FILE):
        with open(MACHINE_ID_FILE, 'r') as f:
            return f.read().strip()
    else:
        return "BAAI_AX_AL_001"

class FlaskServer:
    def __init__(self):
        # 初始化Flask应用
        self.app = Flask(__name__)
        self.app.secret_key = 'agilex'  # 暂时密钥
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        CORS(self.app)

        self.web = "http://120.92.91.171:30083/api"

        #self.web = "http://172.16.17.253:8080/api"
        self.session = requests.Session() 
        self.token = None


        # 初始化日志
        self.init_logging()
        
        # 初始化实例变量
        self.robot_sid = None
        self.video_list = {}
        self.video_timestamp = time.time()
        self.video_streams = {}
        self.stream_status = {}
        self.frame_lock = threading.Lock()
        self.upload_lock = threading.Lock()  # 用于保护 self.upload_nas_flag 的访问
        self.init_streams_flag = False
        self.task_steps = {}
        self.upload_thread = threading.Thread(target=self.time_job, daemon=True)

        self.upload_nas_flag = False
        

        
        # 响应模板
        self.response_start_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        self.response_finish_collection = {
            "timestamp": time.time(),
            "msg": None,
            "data": None
        }
        self.response_submit_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        self.response_discard_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        
        # 注册路由
        self.register_routes()
    
    # --------------------------------定时任务-------------------------------------------------
    def login(self):
        """发送登录请求"""
        url = f"{self.web}/login"  # 假设登录接口是 /login
        
        # 准备请求数据
        # data = {
        #     "username": get_machine_id(),
        #     "password": password
        # }

        data = {

            "username": "eai_data_collect",
            "password": "eai_collect@2025"

        }
        
        try:
            # 发送 POST 请求
            response = self.session.post(url, json=data)
            
            # 检查响应状态码
            if response.status_code == 200:
                # 解析 JSON 响应
                response_data = response.json()

                print("登录云平台成功:", response_data)
                self.token = response_data["token"]
                return True
            else:
                print(f"登录云平台失败，状态码: {response.status_code}, 响应:")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"请求云平台异常: {e}")
            return False
        
    def make_request_with_token(self, path, data):
        """发送带有 token 和请求体的请求"""
        if not self.token:
            print("未登录，无法发送请求")
            return None
    
        url = f"{self.web}/{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",  # 假设使用 Bearer 认证方案
            "Content-Type": "application/json"  # 确保服务器知道我们发送的是 JSON
        }
    
        try:
            if data:
                print(data)
                response = self.session.post(url, headers=headers, json=data)
            else:
                response = self.session.get(url, headers=headers)
            if response.status_code == 200:
                print('---------------------------------------')
                print(f"请求成功，响应：{response.json()}")

                return response.json()
            else:
                print(f"请求失败，状态码: {response.status_code}, 响应: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {e}")
            return None
 
    def local_to_nas(self):
        print(f"任务执行于: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        #time.sleep(5)
        if self.login():
            upload_to_nas.upload()
            with self.upload_lock:
                self.upload_nas_flag = False
        else:
            with self.upload_lock:
                self.upload_nas_flag = False

    def time_job(self):
        schedule.every().day.at("20:00").do(self.local_to_nas)
        print("定时任务已启动，每天20:00执行...")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            print("定时任务已停止")
    
    def init_logging(self):
        """初始化日志配置"""
        now = datetime.datetime.now()
        file_name = "./log/" + now.strftime("%Y.%m.%d.%H.%M") + ".log"
        log_dir = "./log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logging.basicConfig(
            filename=file_name,
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode="a"
        )
    
    def register_routes(self):
        """注册所有路由"""
        # 系统信息
        self.app.add_url_rule('/api/info', 'system_info', self.system_info, methods=['GET'])
        
        # 视频流管理
        self.app.add_url_rule('/api/stream_info', 'stream_info', self.stream_info, methods=['GET'])
        self.app.add_url_rule('/api/get_streams', 'get_streams', self.get_streams, methods=['GET'])
        self.app.add_url_rule('/api/start_stream', 'start_stream', self.start_stream, methods=['POST'])
        self.app.add_url_rule('/api/get_stream/<stream_id>', 'stream_video', self.stream_video, methods=['GET'])
        self.app.add_url_rule('/api/stop_stream/<stream_id>', 'stop_stream', self.stop_stream, methods=['POST'])
        
        # 采集控制
        self.app.add_url_rule('/api/start_collection', 'start_collection', self.start_collection, methods=['POST'])
        self.app.add_url_rule('/api/finish_collection', 'finish_collection', self.finish_collection, methods=['POST'])
        self.app.add_url_rule('/api/discard_collection', 'discard_collection', self.discard_collection, methods=['POST'])
        self.app.add_url_rule('/api/submit_collection', 'submit_collection', self.submit_collection, methods=['POST'])

        # 手动上传nas
        self.app.add_url_rule('/api/manual_upload_nas', 'manual_upload_nas', self.manual_upload_nas, methods=['POST']) 

        # nas上传反馈
        self.app.add_url_rule('/api/upload_start', 'upload_start', self.upload_start, methods=['POST'])
        self.app.add_url_rule('/api/upload_finish', 'upload_finish', self.upload_finish, methods=['POST'])
        self.app.add_url_rule('/api/upload_fail', 'upload_fail', self.upload_fail, methods=['POST'])
        self.app.add_url_rule('/api/upload_process', 'upload_process', self.upload_process, methods=['POST'])
       
        
        # 机器人接口
        self.app.add_url_rule('/robot/update_stream/<stream_id>', 'update_frame', self.update_frame, methods=['POST'])
        self.app.add_url_rule('/robot/stream_info', 'robot_get_video_list', self.robot_get_video_list, methods=['POST'])
        self.app.add_url_rule('/robot/response', 'robot_response', self.robot_response, methods=['POST'])
        self.app.add_url_rule('/robot/get_task_steps', 'get_task_steps', self.get_task_steps, methods=['GET'])
        
        # WebSocket事件
        self.socketio.on_event('connect', self.handle_connect)
        self.socketio.on_event('HEARTBEAT', self.handle_heartbeat)
        self.socketio.on_event('disconnect', self.handle_disconnect)
    
    def send_message_to_robot(self, sid, message):
        """向特定机器人客户端发送消息"""
        self.socketio.emit('robot_command', message, room=sid, namespace='/')

    
    # ---------------------- WebSocket 事件处理 ----------------------
    def handle_connect(self):
        self.robot_sid = request.sid
        print('Client connected')
    
    def handle_heartbeat(self):
        """响应心跳包"""
        emit('HEARTBEAT_RESPONSE', {'server': 'alive'})
    
    def handle_disconnect(self):
        print('Client disconnected')
    
    # ---------------------- 路由处理方法 ----------------------
    def system_info(self):
        """获取系统信息"""
        active_count = sum(1 for s in self.stream_status.values() if s["active"])
        
        return jsonify({
            "status": "running",
            "streams_active": active_count,
            "total_streams": len(self.stream_status),
            "timestamp": time.time(),
            "streams": self.stream_status
        })
    
    def stream_info(self):
        """获取可用视频流列表"""
        status = self.stream_status.copy()
        streams = [
            {"id": sid, "name": info["name"]} 
            for sid, info in status.items()
        ]
        print(f"sget_treams:{streams}")
        return jsonify({"streams": streams})
    
    def get_streams(self):
        try:
            response_data = {
                        "code": 200,
                        "data":self.video_list,
                        "msg":"success"
                    }
            return jsonify(response_data), 200
            #self.send_message_to_robot(self.robot_sid, message={'cmd': 'video_list'})
            """获取可用视频流列表"""
            # now_time = time.time()
            # while True:
            #     if 0 < self.video_timestamp - now_time < 2:
            #         response_data = {
            #             "code": 200,
            #             "data":self.video_list,
            #             "msg":"success"
            #         }
            #         return jsonify(response_data), 200
            #     else:
            #         time.sleep(0.02)
            #     if time.time() - now_time > 5: # 正式环境设为2.5超时
            #         response_data = {
            #             "code": 404,
            #             "data":{},
            #             "msg":"机器人响应超时"
            #         }
            #         return jsonify(response_data), 404          
        except Exception as e:
            response_data = {
                    "code": 500,
                    "data":{},
                    "msg":str(e)
                }
            return jsonify(response_data), 500
    
    def start_stream(self):
        try:
            """启动指定视频流"""
            data = request.get_json()
            stream_id = data.get('stream_id')
            if stream_id not in self.video_streams:
                response_data = {
                        "code": 404,
                        "data":{},
                        "msg":"无效的视频流ID"
                    }

                return jsonify(response_data), 200

                
            success = self.video_streams[stream_id].start()
            if success:
                self.stream_status[stream_id]["active"] = True
                response_data = {
                        "code": 200,
                        "data":{},
                        "msg":"success"
                    }
                return jsonify(response_data), 200
            else:
                response_data = {

                    "code": 404,
                    "data":{},
                    "msg":"启动视频流失败"
                }
                return jsonify(response_data), 200

        except Exception as e:
            response_data = {
                    "code": 500,
                    "data":{},
                    "msg":str(e)
                }
            return jsonify(response_data), 500
    
    def stream_video(self, stream_id):
        try:
            # 尝试将 stream_id 转换为整数
            stream_id = int(stream_id)
        except ValueError:
            response_data = {
                    "code": 400,
                    "data":{},
                    "msg":"无效的流ID,必须为数字"
                }

            return jsonify(response_data), 200

        
        if stream_id not in self.video_streams:
            response_data = {
                    "code": 404,
                    "data":{},
                    "msg":"视频流不存在"
                }

            return jsonify(response_data), 200

      
        if not self.video_streams[stream_id].running:
            response_data = {
                    "code": 404,
                    "data":{},
                    "msg":"视频流未开启"
                }

            return jsonify(response_data), 200

        
        def generate():
            max_retries = 10  # 最大重试次数
            retry_count = 0
            try:
                while True:
                    frame = self.video_streams[stream_id].get_frame()
                    if frame:
                        retry_count = 0  # 重置重试计数器
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"[WARNING] 视频流 {stream_id} 无法获取帧，超过最大重试次数")
                            break  # 退出循环
                    time.sleep(0.03)  # 控制帧率
            except GeneratorExit:
                print(f"[INFO] 客户端断开视频流: {stream_id}")
    
        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    
    def stop_stream(self, stream_id):
        try:
            # 尝试将 stream_id 转换为整数
            stream_id = int(stream_id)
        except ValueError:
            response_data = {
                    "code": 400,
                    "data":{},
                    "msg":"无效的流ID,必须为数字"
                }

            return jsonify(response_data), 200

        
        """停止指定视频流"""
        if stream_id not in self.video_streams:
            response_data = {
                    "code": 404,
                    "data":{},
                    "msg":"视频流不存在"
                }

            return jsonify(response_data), 200

            
        self.video_streams[stream_id].stop()
        self.stream_status[stream_id]["active"] = False
        
        response_data = {
                "code": 200,
                "data":{},
                "msg":"stopped"
            }
        return jsonify(response_data),200
    
    def init_streams(self):
        """初始化视频流"""
        print("in init_streams")

        if 'streams' in self.video_list:
            for stream in self.video_list['streams']:
                self.video_streams[stream['id']] = VideoStream(stream['id'], stream['name'])
                self.stream_status[stream['id']] = {
                    "name": str(stream['name']),
                    "active": False
                }
        print(self.video_list)
    
    def start_collection(self):
        try:
            data = request.get_json()

            data['machine_id'] = get_machine_id()

            self.task_steps = data
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'start_collection','msg': data})
            while True:
                if 0 < self.response_start_collection["timestamp"] - now_time < 2:
                    if self.response_start_collection['msg'] == "success":
                        response_data = {
                            "code": 200,
                            "data":{},
                            "msg":self.response_start_collection['msg']
                        }
                        return jsonify(response_data), 200
                    else:
                        response_data = {
                            "code": 404,
                            "data":{},
                            "msg":self.response_start_collection['msg']
                        }
                        return jsonify(response_data), 404
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 5: # 正式环境设为2.5超时
                    response_data = {
                            "code": 404,
                            "data":{},
                            "msg":"机器人响应超时"
                        }

                    return jsonify(response_data), 200


        except Exception as e:
            response_data = {
                    "code": 500,
                    "data":{},
                    "msg":str(e)
                }
            return jsonify(response_data), 500
    
    def finish_collection(self):
        try:
            data = request.get_json()
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'finish_collection'})
            while True:
                if 0 < self.response_finish_collection["timestamp"] - now_time < 60:
                    if self.response_finish_collection['msg'] == "success":
                        response_data = {
                            "code": 200,
                            "data":self.response_finish_collection['data'],
                            "msg":self.response_finish_collection['msg']
                        }
                        return jsonify(response_data), 200
                    else:
                        response_data = {
                            "code": 404,
                            "data":{},
                            "msg":self.response_finish_collection['msg']
                        }

                        return jsonify(response_data), 200
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 70: # 正式环境设为2.5超时

                    response_data = {
                            "code": 404,
                            "data":{},
                            "msg":"机器人响应超时"
                        }

                    return jsonify(response_data), 200


        except Exception as e:
            response_data = {
                    "code": 500,
                    "data":{},
                    "msg":str(e)
                }
            return jsonify(response_data), 500
        
    def discard_collection(self):
        try:
            data = request.get_json()
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'discard_collection'})
            while True:
                if 0 < self.response_discard_collection["timestamp"] - now_time < 3:
                    if self.response_discard_collection['msg'] == "success":
                        response_data = {
                            "code": 200,
                            "data":{},
                            "msg":"success"
                        }
                        return jsonify(response_data), 200
                    else:
                        response_data = {
                            "code": 404,
                            "data":{},
                            "msg":self.response_discard_collection['msg']
                        }

                        return jsonify(response_data), 200

                else:
                    time.sleep(0.02)
                if time.time() - now_time > 5: # 正式环境设为2.5超时
                    response_data = {
                            "code": 404,
                            "data":{},
                            "msg":"机器人响应超时"
                        }

                    return jsonify(response_data), 200


        except Exception as e:
            response_data = {
                    "code": 500,
                    "data":{},
                    "msg":str(e)
                }
            return jsonify(response_data), 500
        
    def submit_collection(self):
        try:
            data = request.get_json()
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'submit_collection'})
            while True:
                if 0 < self.response_submit_collection["timestamp"] - now_time < 3:
                    if self.response_submit_collection['msg'] == "success":
                        response_data = {
                            "code": 200,
                            "data":{},
                            "msg":"success"
                        }
                        return jsonify(response_data), 200
                    else:
                        response_data = {
                            "code": 404,
                            "data":{},
                            "msg":self.response_submit_collection['msg']
                        }

                        return jsonify(response_data), 200

                else:
                    time.sleep(0.02)
                if time.time() - now_time > 5: # 正式环境设为2.5超时
                    response_data = {
                            "code": 404,
                            "data":{},
                            "msg":"机器人响应超时"
                        }

                    return jsonify(response_data), 200


        except Exception as e:
            response_data = {
                    "code": 500,
                    "data":{},
                    "msg":str(e)
                }
            return jsonify(response_data), 500
    
    def standby(self):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        pass

    def manual_upload_nas(self):
        try:
             with self.upload_lock:  # 使用 upload_lock 来保护 self.upload_nas_flag
                if self.upload_nas_flag:
                    response_data = {
                        "code": 401,
                        "data": {},
                        "msg": '数据上传中'
                    }
                    return jsonify(response_data), 200
                else:
                    self.upload_nas_flag = True
                    upload_manual_thread = threading.Thread(target=self.local_to_nas,daemon=True)
                    upload_manual_thread.start()
                    response_data = {
                        "code": 200,
                        "data": {},
                        "msg": 'success'
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            response_data = {
                    "code": 500,
                    "data":{},
                    "msg":str(e)
                }
            return jsonify(response_data), 500

    # ---------------------------------------upload----------------------------------------------
    def upload_start(self):
        try:
            data = request.get_json()
            data['transfer_type'] = 'local_to_nas'
            self.make_request_with_token('eai/dts/upload/start',data)
            return jsonify({}), 200
        except Exception as e:
            return jsonify({'error':str(e)}), 500

    def upload_finish(self):
        try:
            data = request.get_json()
            response_data = {
                    "task_id": data["task_id"],                 
                    "task_data_id": data["task_data_id"],           
                    "transfer_type": "local_to_nas" ,     
                    "status" : "SUCCESS" 
                }
            self.make_request_with_token('eai/dts/upload/complete',response_data)
            return jsonify({}), 200
        except Exception as e:
            return jsonify({'error':str(e)}), 500
    
    def upload_fail(self):
        try:
            data = request.get_json()
            if data.get("expand"):
                response_data = {
                    "task_id": data["task_id"],                 
                    "task_data_id": data["task_data_id"],           
                    "transfer_type": "local_to_nas" ,     
                    "status" : "FAILED",
                    "expand": data['expand'] 
                }
            else:
                response_data = {
                        "task_id": data["task_id"],                 
                        "task_data_id": data["task_data_id"],           
                        "transfer_type": "local_to_nas" ,     
                        "status" : "FAILED",
                        "expand": '{"nas_failed_msg":"网络通讯错误"}' 
                    }
            self.make_request_with_token('eai/dts/upload/complete',response_data)
            return jsonify({}), 200
        except Exception as e:
            return jsonify({'error':str(e)}), 500

    def upload_process(self):
        try:
            data = request.get_json()
            data['transfer_type'] = 'local_to_nas'
            self.make_request_with_token('eai/dts/upload/process',data)
            return jsonify({}), 200
        except Exception as e:
            return jsonify({'error':str(e)}), 500

    # ---------------------------------------robot------------------------------------------------
    def update_frame(self, stream_id):
        try:
            # 尝试将 stream_id 转换为整数
            stream_id = int(stream_id)
        except ValueError:
            return jsonify({"error": "无效的流ID,必须为数字"}), 400

        if stream_id not in self.video_streams:
            logging.error(f"Invalid stream ID: {stream_id}")
            return jsonify({"error": "无效的视频流ID"}), 401

        frame_data = request.get_data()
        if not frame_data:
            logging.error("No frame data received")
            return jsonify({"error": "未接收到帧数据"}), 402
        
        try:
            # 可选：验证是否为 JPEG 数据
            img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid JPEG data")
        except Exception as e:
            logging.error(f"Invalid frame data: {e}")
            return jsonify({"error": "无效的帧数据"}), 404

        self.video_streams[stream_id].update_frame(frame_data)
        return jsonify({"msg": "帧已更新"}), 200
    
    def robot_get_video_list(self):
        try:

            new_list = request.get_json()

            self.video_list = json.loads(new_list)
            print(f"self.video_list: {self.video_list}")

            self.video_timestamp = time.time()
            if not self.init_streams_flag:
                self.init_streams()
                self.init_streams_flag = True
            return jsonify({}), 200
        except Exception as e:
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def robot_response(self):
        try:
            data = request.get_json()
            print('-----------------')
            print(data)
            if data["cmd"] == "start_collection":
                self.response_start_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            elif data["cmd"] == "finish_collection":
                self.response_finish_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"],
                    "data": data["data"]
                }
            elif data["cmd"] == "discard_collection":
                self.response_discard_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            elif data["cmd"] == "submit_collection":
                self.response_submit_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            return jsonify({}), 200
        except Exception as e:
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def get_task_steps(self):
        return jsonify(self.task_steps), 200
    
    def run(self):
        self.upload_thread.start()
        """运行服务器"""
        self.socketio.run(self.app, host='0.0.0.0', port=8080, debug=False)



if __name__ == '__main__':
    server = FlaskServer()
    server.run()
