import network
import urequests
import json
import gc
import ubinascii

# 连接WiFi
sta_if = network.WLAN(network.STA_IF)
sta_if.active(True)
sta_if.connect("Your_Wifi_Name", "Your_Wifi_Password")
while not sta_if.isconnected():
    pass
print("Connected! IP:", sta_if.ifconfig()[0])

# DeepSeek API配置
API_KEY = "sk-4******************28"  # 请使用有效密钥
API_URL = "https://api.deepseek.com/v1/chat/completions"

def ask_llm(prompt):
    # 创建请求头
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json; charset=utf-8"  # 添加字符集声明
    }
    
    # 创建请求体
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        print("\nSending request to DeepSeek API...")
        
        # 手动构建JSON字符串（确保UTF-8编码）
        json_payload = '{"model":"deepseek-chat","messages":[{"role":"user","content":"'
        
        # 安全转义中文字符
        escaped_prompt = ""
        for char in prompt:
            if char == '"':
                escaped_prompt += '\\"'
            elif char == '\\':
                escaped_prompt += '\\\\'
            else:
                escaped_prompt += char
        
        json_payload += escaped_prompt
        json_payload += '"}],"max_tokens":100,"temperature":0.7}'
        
        print("JSON Payload:", json_payload)
        print("Length:", len(json_payload))
        
        # 发送请求前检查内存
        gc.collect()
        free_mem = gc.mem_free()
        alloc_mem = gc.mem_alloc()
        print(f"Memory: Free={free_mem}, Allocated={alloc_mem}")
        
        # 发送请求（显式编码为UTF-8）
        response = urequests.post(
            API_URL, 
            headers=headers, 
            data=json_payload.encode('utf-8')  # 关键修改：显式编码
        )
        
        status = response.status_code
        raw_response = response.text
        
        print(f"API Response Status: {status}")
        print("Raw Response:", raw_response[:200])  # 打印前200个字符
        
        # 检查HTTP状态码
        if status != 200:
            return f"API error {status}: {raw_response[:100]}"  # 返回错误摘要
        
        # 解析JSON响应
        try:
            response_data = json.loads(raw_response)
            
            # 提取助手的回复内容
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                return "Unexpected API response format"
                
        except ValueError as e:
            print("JSON parse error:", e)
            return "JSON parse error in API response"
            
    except Exception as e:
        print("Request failed:", e)
        return f"API request failed: {str(e)}"

# 使用示例
print("\n=== DeepSeek API 测试 ===")

# 测试中文问题 - 使用简单中文
question = "什么是MicroPython?"
print(f"\n提问: {question}")
response = ask_llm(question)
print(f"\nAI回复: {response}")

# 测试更复杂的中文问题
question = "ESP32和MicroPython有什么关系？"
print(f"\n提问: {question}")
response = ask_llm(question)
print(f"\nAI回复: {response}")