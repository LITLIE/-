import time
import camera
import network
import uasyncio as asyncio
from machine import Pin, SPI, I2C, reset
import atk_xl9555 as io_ex
import atk_lcd as lcd
import urequests
import json
import gc
import ubinascii

# --------- 硬件初始化 ----------
def hardware_init():
    try:
        i2c0 = I2C(0, scl=Pin(42), sda=Pin(41), freq=400000)
        xl9555 = io_ex.init(i2c0)
        
        # 初始化蜂鸣器（低电平触发）
        xl9555.write_bit(io_ex.BEEP, 0)
        
        # 摄像头复位序列
        xl9555.write_bit(io_ex.OV_RESET, 0)  # 复位拉低
        xl9555.write_bit(io_ex.OV_PWDN, 1)  # 电源关闭
        time.sleep_ms(150)
        
        xl9555.write_bit(io_ex.OV_RESET, 1)  # 复位释放
        xl9555.write_bit(io_ex.OV_PWDN, 0)  # 电源开启
        time.sleep_ms(250)  # 等待摄像头稳定
        
        # LCD 复位序列
        xl9555.write_bit(io_ex.SLCD_RST, 0)
        time.sleep_ms(100)
        xl9555.write_bit(io_ex.SLCD_RST, 1)
        time.sleep_ms(150)
        
        # 初始化 SPI 和 LCD
        spi = SPI(2, baudrate=80000000, sck=Pin(12), mosi=Pin(11), miso=Pin(13))
        display = lcd.init(
            spi,
            dc=Pin(40, Pin.OUT, Pin.PULL_UP, value=1),
            cs=Pin(21, Pin.OUT, Pin.PULL_UP, value=1),
            dir=1, 
            lcd=0
        )
        xl9555.write_bit(io_ex.SLCD_PWR, 1)  # LCD电源开启
        time.sleep_ms(200)
        
        # 摄像头初始化（带重试机制）
        cam_success = False
        for i in range(5):
            try:
                cam = camera.init(
                    0, 
                    format=camera.JPEG, 
                    fb_location=camera.PSRAM,
                    framesize=camera.FRAME_QQVGA,  # 使用更小的分辨率
                    xclk_freq=20000000
                )
                if cam:
                    print(f"Camera initialized on attempt {i+1}")
                    cam_success = True
                    break
            except Exception as e:
                print(f"Camera init error: {e}")
            camera.deinit()
            time.sleep(1)
        
        if not cam_success:
            print("Camera init failed after retries")
            return None, None
        
        return xl9555, display
    
    except Exception as e:
        print(f"Hardware init failed: {e}")
        return None, None

# --------- 连接 WiFi ----------
def connect_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.disconnect()  # 确保从之前的连接中断开
    
    print(f'Connecting to {ssid}...')
    wlan.connect(ssid, password)
    
    max_retry = 20
    for i in range(max_retry):
        if wlan.isconnected():
            break
        print(f'Waiting... ({i+1}/{max_retry})')
        time.sleep(1)
    
    if not wlan.isconnected():
        print('WiFi connection failed!')
        return None
    
    print('Network config:', wlan.ifconfig())
    return wlan.ifconfig()[0]

# Moonshot API配置
MOONSHOT_API_KEY = "sk-1K7*******************ssY"  # 替换为你的Moonshot API密钥，也可利用其他模型
MOONSHOT_API_URL = "https://api.moonshot.cn/v1/chat/completions" # 替换为你所需要使用的模型

def analyze_image_with_ai(image_data, prompt="请描述这张图片的内容"):
    """
    使用Moonshot API分析图片
    :param image_data: 图片的二进制数据
    :param prompt: 给AI的提示词
    :return: AI的分析结果
    """
    # 将图片转换为base64编码
    image_b64 = ubinascii.b2a_base64(image_data).decode('utf-8').strip()
    
    # 创建请求头
    headers = {
        "Authorization": f"Bearer {MOONSHOT_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 创建Moonshot格式的请求体
    payload = {
        "model": "moonshot-v1-8k-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    
    try:
        print("\nSending image to Moonshot API...")
        print(f"Image size: {len(image_data)} bytes, Base64 size: {len(image_b64)} chars")
        
        # 发送请求前检查内存
        gc.collect()
        free_mem = gc.mem_free()
        alloc_mem = gc.mem_alloc()
        print(f"Memory: Free={free_mem}, Allocated={alloc_mem}")
        
        # 发送请求
        response = urequests.post(
            MOONSHOT_API_URL, 
            data=json.dumps(payload).encode('utf-8'),
            headers=headers
        )
        
        status = response.status_code
        raw_response = response.text
        
        print(f"API Response Status: {status}")
        print("Raw Response:", raw_response[:200])  # 打印前200个字符
        
        # 检查HTTP状态码
        if status != 200:
            # 尝试解析错误信息
            try:
                error_data = json.loads(raw_response)
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
                return f"API error {status}: {error_msg}"
            except:
                return f"API error {status}: {raw_response[:200]}"
        
        # 解析JSON响应
        response_data = json.loads(raw_response)
        
        # 提取助手的回复内容
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            return "Unexpected API response format"
            
    except Exception as e:
        print("Image analysis request failed:", e)
        # 打印详细错误信息
        import sys
        sys.print_exception(e)
        return f"API request failed: {str(e)}"
    finally:
        # 确保关闭响应以释放内存
        if 'response' in locals():
            response.close()
        gc.collect()


# 添加 URL 解码函数
def urldecode(encoded_str):
    """
    简单的 URL 解码实现
    :param encoded_str: 编码后的字符串（字节形式）
    :return: 解码后的字符串（UTF-8）
    """
    decoded_bytes = bytearray()
    i = 0
    while i < len(encoded_str):
        if encoded_str[i] == 37:  # 37 是 '%' 的 ASCII 码
            # 处理 %XX 形式的编码
            if i + 2 < len(encoded_str):
                hex_str = encoded_str[i+1:i+3]
                try:
                    decoded_bytes.append(int(hex_str, 16))
                    i += 3
                    continue
                except:
                    # 转换失败，保留原字符
                    pass
        elif encoded_str[i] == 43:  # 43 是 '+' 的 ASCII 码
            # 将 '+' 转换为空格
            decoded_bytes.append(32)
            i += 1
            continue
        
        decoded_bytes.append(encoded_str[i])
        i += 1
    
    return decoded_bytes.decode('utf-8', 'ignore')


# --------- HTTP 服务器 ----------
async def handle_client(reader, writer):
    try:
        request = await reader.read(1024)
        client_ip = writer.get_extra_info('peername')[0]
        print(f"\nRequest from {client_ip}")
        
        # 捕获图像请求
        if b"GET /capture" in request:
            # 捕获图像
            buf = camera.capture()
            if not buf:
                error_response = "HTTP/1.1 500 Internal Server Error\r\n"
                error_response += "Content-Type: text/plain; charset=utf-8\r\n"
                error_response += "Connection: close\r\n\r\n"
                error_response += "Camera capture failed"
                await writer.awrite(error_response.encode('utf-8'))
                return

            # 返回JPEG图像
            headers = "HTTP/1.1 200 OK\r\n"
            headers += "Content-Type: image/jpeg\r\n"
            headers += "Connection: close\r\n"
            headers += "Content-Length: {}\r\n\r\n".format(len(buf))
            await writer.awrite(headers.encode() + buf)
            print(f"Sent image to {client_ip}")

        # AI分析图像请求
        elif b"GET /analyze" in request:
            # 从请求中提取提示词
            prompt = "请描述这张图片的内容"
            if b"prompt=" in request:
                try:
                    # 提取类似: GET /analyze?prompt=描述图片中的物体
                    start_idx = request.index(b"prompt=") + 7
                    end_idx = request.index(b" HTTP/1.1")  # 注意这里添加了空格
                    prompt_bytes = request[start_idx:end_idx].split(b"&")[0]
                    prompt = urldecode(prompt_bytes)
                    print(f"Using custom prompt: {prompt}")
                except Exception as e:
                    print(f"Prompt extraction error: {e}")
            
            # 捕获图像
            buf = camera.capture()
            
            if not buf:
                error_response = "HTTP/1.1 500 Internal Server Error\r\n"
                error_response += "Content-Type: text/plain; charset=utf-8\r\n"
                error_response += "Connection: close\r\n\r\n"
                error_response += "Camera capture failed"
                await writer.awrite(error_response.encode('utf-8'))
                return
            
            print(f"Analyzing image for {client_ip}... Size: {len(buf)} bytes")
            
            # 使用AI分析图像
            analysis_result = analyze_image_with_ai(buf, prompt)
            
            # 返回分析结果
            headers = "HTTP/1.1 200 OK\r\n"
            headers += "Content-Type: application/json; charset=utf-8\r\n"
            headers += "Connection: close\r\n\r\n"
            
            # 创建JSON响应
            response_data = {
                "status": "success" if "API error" not in analysis_result else "error",
                "analysis": analysis_result,
                "image_size": len(buf),
                "prompt": prompt
            }
            
            # 修复：确保正确编码响应数据
            json_data = json.dumps(response_data)
            await writer.awrite(headers.encode('utf-8') + json_data.encode('utf-8'))
            print(f"Sent analysis to {client_ip}")
        
        # 主页面请求
        else:
            # 返回简单主页 - 明确指定UTF-8编码
            html = """HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ESP32 AI 相机</title>
  <!-- 引用 Exo 2 字体 -->
  <link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@400;700&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { width:100%; height:100%; }

    body {
      font-family: 'Exo 2', sans-serif;
      background: #e0e5ec;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
    }

    .container {
      width: 90%;
      max-width: 800px;
    }

    /* 通用卡片风格 + Neumorphism */
    .card {
      width: 90%;
      max-width: 800px;
      margin: 1rem auto;  /* 所有卡片都居中 */
    }

    .card {
      background: #e0e5ec;
      border-radius: 16px;
      box-shadow:
        8px 8px 16px rgba(163,177,198,0.6),
       -8px -8px 16px rgba(255,255,255,0.8);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      padding: 25px;
      margin-bottom: 25px;
    }
    .card:hover {
      transform: translateY(-4px);
      box-shadow:
        12px 12px 24px rgba(163,177,198,0.6),
       -12px -12px 24px rgba(255,255,255,0.8);
    }
    button {
        padding: 12px 24px;
        background: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s;
        display: block;
        margin: 15px auto 0;
}

    header.card h1 {
      font-size: 2.5rem;
      font-weight: 700;
      color: #333;
      text-align: center;
      margin-bottom: 0.5rem;
    }
    header.card h2 {
      font-size: 1.25rem;
      font-weight: 400;
      color: #666;
      text-align: center;
    }
    
    /* 给视频卡片留出内边距 */
    .video-container.card {
      padding: 20px;    /* 原来是 padding:0; 现在改为 20px */
      overflow: hidden;
    }

    /* 确保图片在卡片内有边距 */
    .video-container.card img {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 12px;
      margin: 0 auto;  /* 图片居中 */
    }

    .divider {
      width: 70%;
      height: 1px;
      background: rgba(0,0,0,0.1);
      margin: 1rem auto;
      border-radius: 1px;
    }

    .video-container.card {
      padding: 0;
      overflow: hidden;
    }
    .video-container.card img {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 16px;
    }

    .analysis.card label {
      display: block;
      margin-bottom: 8px;
      color: #333;
    }
    .analysis.card input {
      width: 100%;
      padding: 12px;
      border: 1px solid #ddd;
      border-radius: 6px;
      font-size: 16px;
      margin-bottom: 15px;
      box-sizing: border-box;
    }

    .analysis.card #result {
      margin-top: 20px;
      padding: 20px;
      background: #e8f4fd;
      border-radius: 8px;
      white-space: pre-wrap;
      line-height: 1.6;
      border-left: 4px solid #3498db;
      color: #333;
    }

    .status {
      text-align: center;
      margin-bottom: 30px;
      color: #7f8c8d;
      font-size: 14px;
    }
    
    footer.card div {
      color: #444;
      font-size: 0.95rem;
      margin: 0.5rem 0;
      text-align: center;
    }
    
    @media (max-width: 600px) {
      header.card h1 { font-size: 2rem; }
      header.card h2 { font-size: 1rem; }
      .divider { width: 90%; }
    }
  </style>
</head>
<body>

  <!-- 独立 Header 卡片 -->
  <header class="card">
    <h1>ESP32 AI 视觉系统</h1>
    <h2>Course: Intelligent Embedded Vision Applications</h2>
  </header>

  <div class="divider"></div>

  <div class="container">
    <div class="card video-container">
      <h2 style="text-align:center; color:#3498db; margin-bottom:15px;">实时图像</h2>
      <img id="live-image" src="/capture"
           onerror="this.src='data:image/svg+xml;charset=UTF-8,<svg xmlns=&quot;http://www.w3.org/2000/svg&quot; viewBox=&quot;0 0 400 300&quot;><rect width=&quot;400&quot; height=&quot;300&quot; fill=&quot;%23f0f2f5&quot;/><text x=&quot;50%&quot; y=&quot;50%&quot; font-family=&quot;Arial&quot; font-size=&quot;16&quot; fill=&quot;%23999&quot; text-anchor=&quot;middle&quot; dominant-baseline=&quot;middle&quot;>正在加载图像...</text></svg>';" />
      <button onclick="refreshImage()">刷新图像</button>
    </div>

    <div class="divider"></div>

    <div class="card analysis">
      <h2 style="text-align:center; color:#3498db; margin-bottom:15px;">AI 图像分析</h2>
      <label for="prompt-input">分析提示:</label>
      <input type="text" id="prompt-input" name="prompt"
             value="请描述这张图片的内容" placeholder="输入分析提示..." />
      <button onclick="analyzeImage()">分析图像</button>
      <div id="result">等待分析结果...</div>
    </div>

    <div class="status">
      <p>设备 IP: <span id="device-ip">加载中...</span></p>
      <p>状态: <span id="status">就绪</span></p>
    </div>

    <div class="divider"></div>

<!--     <div class="card">
#       <div>Team Members: Lu Yiming | Chen Litian | Huang Xuankai | Ou Peiyi | Qian Yibin | Wang Zichang</div>
#       <div>Instructors: He Hui &amp; Liu Chunxiu</div>
#       <div>Sponsors: Turinger</div>
#     </div>
-->
    <footer class="card">
        <div>© 2025 Rubbish Vision Tracking Project</div>
        <div class="members">Team Members: Lu Yiming | Chen Litian | Huang Xuankai | Ou Peiyi | Qian Yibin | Wang Zichang</div>
        <div class="instructor">Instructors: He Hui &amp; Liu Chunxiu</div>
        <div class="sponsors">Sponsors: Turinger</div>
      </footer>
  </div>

  <script>
    function getDeviceIP() {
      document.getElementById('device-ip').textContent = window.location.hostname;
    }
    function refreshImage() {
      const img = document.getElementById('live-image');
      img.src = '/capture?' + Date.now();
      document.getElementById('status').textContent = '图像已刷新';
    }
    function analyzeImage() {
      const prompt = document.getElementById('prompt-input').value;
      const resultDiv = document.getElementById('result');
      const statusEl = document.getElementById('status');
      resultDiv.innerHTML = '<div class="loading">分析中，请稍候...</div>';
      statusEl.textContent = '正在分析图像...';
      const timeoutId = setTimeout(() => {
        resultDiv.innerHTML = '<div class="loading">分析时间较长，请耐心等待...</div>';
      }, 3000);
      fetch('/analyze?prompt=' + encodeURIComponent(prompt))
        .then(resp => { clearTimeout(timeoutId); if (!resp.ok) throw new Error(resp.status); return resp.json(); })
        .then(data => {
          resultDiv.innerHTML = data.status==='success' ? data.analysis : '分析失败: '+(data.message||data.analysis);
          statusEl.textContent = data.status==='success' ? '分析完成' : '分析失败';
        })
        .catch(err => {
          clearTimeout(timeoutId);
          resultDiv.innerHTML = '请求错误: '+err.message;
          statusEl.textContent = '请求出错';
        });
    }
    window.onload = () => { getDeviceIP(); refreshImage(); };
  </script>
</body>
</html>
"""
            await writer.awrite(html.encode('utf-8'))
            print(f"Served homepage to {client_ip}")
    
    except Exception as e:
        print(f"Client handling error: {e}")
        # 打印详细错误信息
        import sys
        sys.print_exception(e)
    finally:
        await writer.wait_closed()
        gc.collect()

async def start_server():
    server = await asyncio.start_server(handle_client, "0.0.0.0", 80)
    print("HTTP server running on port 80")
    while True:
        await asyncio.sleep(5)  # 保持服务器运行

# --------- 主程序入口 ----------
def main():
    # 硬件初始化
    xl9555, display = hardware_init()
    if xl9555 is None or display is None:
        print("Critical hardware failure! Rebooting...")
        time.sleep(5)
        reset()
    
    # 连接WiFi - 增加重试机制
    max_wifi_retries = 3
    ip = None
    for i in range(max_wifi_retries):
        ip = connect_wifi("WiFi_Name", "WiFi_Password")  # 使用你的WiFi凭证
        if ip:
            break
        print(f"WiFi connection failed, retry {i+1}/{max_wifi_retries}")
        time.sleep(5)
    
    if not ip:
        print("WiFi failed after retries! Rebooting...")
        time.sleep(5)
        reset()
    
    print(f"Camera ready at http://{ip}/capture")
    print(f"AI analysis at http://{ip}/analyze")
    print(f"Web interface: http://{ip}")
    
    # 启动服务器
    try:
        asyncio.run(start_server())
    except Exception as e:
        print(f"Server crashed: {e}")
        print("Rebooting...")
        time.sleep(5)
        reset()

# 运行主程序
if __name__ == "__main__":
    main()


