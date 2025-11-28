import pyarrow as pa
import time
from dora import Node


if __name__ == "__main__":

    node = Node()

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] = "actions":
                event["value"].to_numpy()
                buffer_bytes = event["value"].to_numpy().tobytes()
                            
                # 处理接收到的数据
                # print(f"Send event: {event_id}")
                # print(f"Buffer size: {len(buffer_bytes)} bytes")

                try:
                    socket.send_multipart([
                        event_id.encode('utf-8'),
                        buffer_bytes
                    ], flags=zmq.NOBLOCK)
                except zmq.Again:
                    # print("Socket would block, skipping send this frame...")
                    # continue
                    # continue
                    pass
                # time.sleep(0.001)
            
        elif event["type"] == "STOP":
            break