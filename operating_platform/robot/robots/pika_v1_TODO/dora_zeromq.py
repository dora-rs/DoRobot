import zmq
import threading
import pyarrow as pa
import time
from dora import Node
import json

# IPC Address
ipc_address = "ipc:///tmp/dr-robot-pika-v1"

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect(ipc_address)
socket.setsockopt(zmq.SNDHWM, 2000)
socket.setsockopt(zmq.SNDBUF, 100 * 1024 * 1024)
running_server = True


# def recv_server():
#     while running_server:
#         try:
#             # 接收 multipart 消息
#             message_parts = socket.recv_multipart(flags=zmq.NOBLOCK)
#             if message_parts:
#                 # 假设接收到的消息是简单的字符串
#                 message = b'|'.join(message_parts).decode('utf-8')
#                 print("Received:", message)
#         except zmq.Again:
#             time.sleep(0.01)  # 避免忙等
#         except Exception as e:
#             print("recv error:", e)
#             break



if __name__ == "__main__":

    node = Node()

    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]
            buffer_bytes = event["value"].to_numpy().tobytes()
            meta_bytes = json.dumps(event["metadata"]).encode('utf-8')
            # 处理接收到的数据
            # print(f"Send event: {event_id}")
            # print(f"Buffer size: {len(buffer_bytes)} bytes")

            try:
                socket.send_multipart([
                    event_id.encode('utf-8'),
                    buffer_bytes,
                    meta_bytes,
                ], flags=zmq.NOBLOCK)
            except zmq.Again:
                # print("Socket would block, skipping send this frame...")
                # continue
                # continue
                pass
            # time.sleep(0.001)
            
        elif event["type"] == "STOP":
            break
    
    # Close server 
    running_server = False
    # server_thread.join()

    # Close zmq
    socket.close()
    context.term()
