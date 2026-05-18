'''V01版本的agent实现内容：
        1.首先是最重要的agent_loop，这个循环控制程序不断操作，直到LLM的回答没有调用工具(TOOLS)时循环终止。
        2.其次是给LLM提供了一个当前可调用的工具 bash ，一个工具对应一个函数，当LLM决定要使用某个工具时，程序就调用对应工具的函数，实现操作。
        对应关系：bash---run_bash()  '''



import os          # os 库就是 Python 和「操作系统」对话的工具，例如读取环境变量
import subprocess  # 让 Python 执行系统命令（cmd /bash 命令）

from anthropic import Anthropic
from dotenv import load_dotenv  # 这是加载 .env 环境变量的工具

# 加载环境变量.env
load_dotenv(override=True)
client = Anthropic(  # client为大模型
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    default_headers={
        #  强制在每一次请求里，都加上一个 HTTP 请求头，第三方模型
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
)
MODEL = os.getenv("MODEL_ID")

# agent身份描述
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# 工具列表 提供给大模型
TOOLS = [{
    "name":"bash",
    "description":"Run a shell command.",
    "input_schema":{
        "type":"object",
        "properties":{"command":{"type":"string"}},
        "required":["command"],
    },
}]

# 对应工具bash的操作
def run_bash(command:str)->str:   # python语法：输入一个字符串(LLM给的命令)、输出一个字符串(程序执行结果返回给LLM)
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]   # 危险操作列表
    if any(d in command for d in dangerous):
        return "Error:Dangerous command blocked"
    try:
        # 程序运行命令
        r = subprocess.run(command,
                           shell=True,
                           cwd=os.getcwd(),
                           capture_output=True,
                           text=True,
                           timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 返回给LLM结果 太长就截断
        return out[:5000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error:Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error:{e}"

# 循环控制程序不断操作
def agent_loop(messages:list):  # 这里的输入是完整的对话历史记录/上下文 同时也包括用户输入
    while True:
        # 发给LLM生成命令  命令存到response中
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        # 把刚刚LLM回答存到上下文中
        messages.append({"role":"assistant", "content":response.content})
        # 不再调用工具 循环终止
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                # 这里的command是LLM命令  eg:"command": "dir" 这是bash工具里面的工具
                # 因为现在只有bash一个工具  直接传给run_bash
                output = run_bash(block.input['command'])
                print(output[:200])
                results.append({"type":"tool_result","tool_use_id":block.id,"content":output})
        # 把结果加入到上下文
        messages.append({"role":"user", "content":results})


if __name__ == "__main__":
    history = []
    while True:  # 实现用户和agent无限聊天
        try:
            # 真正的用户输入
            query = input("\033[36ms01>>\033[0m")
        except(EOFError, KeyboardInterrupt):  # 用户按ctrl+c可以退出
            break
        if query.strip().lower() in ("q", "exit", ""):  # 输入q/exit 可以退出
            break
        history.append({"role":"user", "content":query})
        agent_loop(history)
        # 取出并显示LLM最后返回的内容
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()