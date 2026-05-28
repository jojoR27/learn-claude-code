'''V08版本的agent实现内容：(基于V02 + V07的threading)
        V08和V04版本的区别---04版本子代理，只有当主agent分派任务之后，函数调用子代理才启动，做完任务立马销毁；  （任务拆分）
                          08版本团队代理，spawn创建代理一直存活在后台线程，threading不阻塞主agent、拥有独立长期的上下文、代理之间可以广播收件相互协作；  （多智能体协作）
        1.MessageBus类：操作代理之间收发消息，相互协作；
        2.TeammateManager类：操作一直存在在config.json的代理团队；

        对应关系：bash---run_bash()
                read_file---run_read()
                write_file---run_write()
                edit_file---run_edit()
                spawn_teammate---TeammateManager.spawn()
                list_teammates---TeammateManager.list_all()
                send_message---MessageBus.send()
                read_inbox---MessageBus.read_inbox()
                broadcast---MessageBus.broadcast()'''



import os          # os 库就是 Python 和「操作系统」对话的工具，例如读取环境变量
import subprocess  # 让 Python 执行系统命令（cmd /bash 命令）
from pathlib import Path  # 用于后面限制文件操作的
import threading
import time
import json

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

# 固定LLM的工作目录 为当前目录
WORKDIR = Path.cwd()
# 团队配置的存放目录  里面有每个agent的.jsonl还有inbox子目录
TEAM_DIR = WORKDIR / ".team"
# 每个agent的收件箱目录
INBOX_DIR = TEAM_DIR / "inbox"

# agent身份prompt描述
SYSTEM =  f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

# 合法消息类型
VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# 消息总线类，每个agent一个jsonl文件为收件箱
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid types are {VALID_MSG_TYPES}"
        # 标准消息结构
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)

        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []

        messages = []
        content = inbox_path.read_text(encoding="utf-8")
        for line in content.strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("", encoding="utf-8")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, msg_type="broadcast")
                count += 1
            return f"Broadcast to {count} teammates"

BUS = MessageBus(INBOX_DIR)

# agent团队管理类
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    # 加载团队配置信息
    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        return {"team_name": "default", "members":[]}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")

    def _find_member(self, name: str) -> dict:
        for member in self.config["members"]:
            if member["name"] == name:
                return member
        return None

    # 创建或启动Agent(config.jsonl)   创建并启动agent线程
    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: Member '{name}' is currently '{member['status']}'"
            member["status"] = "working"
            member["role"] = role
        else:  # 可以新建
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()

        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawning '{name}' with role '{role}'"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        sys_prompt = {
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        }
        # 每个agent有独立的上下文
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        # 循环执行任务
        for _ in range(50):
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg, ensure_ascii=False)})

            try:
                response = client.messages.create(
                    model=MODEL,
                    system=sys_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=8000,
                )
            except Exception:
                break

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = self._exec(name, block.name, block.input)
                    print(f"  [{name}] {block.name}: {str(output)[:120]}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })
            messages.append({"role": "user", "content": results})

        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()

    # agent工具映射
    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        if tool_name == "bash":
            return run_bash(args["command"])
        if tool_name == "read_file":
            return run_read(args["path"])
        if tool_name == "write_file":
            return run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2, ensure_ascii=False)
        return f"Unknown tool: {tool_name}"

    # agent工具列表
    def _teammate_tools(self) -> list:
        return [
            {"name": "bash",
             "description": "Run a shell command.",
             "input_schema":
                 {"type": "object",
                  "properties": {"command": {"type": "string"}},
                  "required": ["command"]}},
            {"name": "read_file",
             "description": "Read file contents.",
             "input_schema":
                 {"type": "object",
                  "properties": {"path": {"type": "string"}},
                  "required": ["path"]}},
            {"name": "write_file",
             "description": "Write content to file.",
             "input_schema":
                 {"type": "object",
                  "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                  "required": ["path", "content"]}},
            {"name": "edit_file",
             "description": "Replace exact text in file.",
             "input_schema":
                 {"type": "object",
                  "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},"new_text": {"type": "string"}},
                  "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message",
             "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"},"msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}},
             "required": ["to", "content"]}},
            {"name": "read_inbox",
             "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
        ]

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]

TEAM = TeammateManager(TEAM_DIR)

# 强制LLM只能在当前目录操作
def safe_path(p:str) -> Path:
    # 把用户输入的路径，拼接成当前工作目录下的绝对路径
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    # 路径安全返回拼接好的路径
    return path

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

# 对应工具read_file
def run_read(path: str, limit: int = None) -> str:  # limit是控制读取行数，默认是全部  输出是文件内容
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        # 如果超出limit限制
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines)-limit} more lines"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error:{e}"

# 对应工具write_file
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        # 文件不存在则创建
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {fp}"
    except Exception as e:
        return f"Error:{e}"

# 对应工具edit_file
def run_edit(path: str, old_content: str, new_content: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_content not in content:
            return f"Error: Content not found in {fp}"
        fp.write_text(content.replace(old_content, new_content, 1))
        return f"Edited {fp}"
    except Exception as e:
        return f"Error:{e}"

# 工具映射表
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_content"], kw["new_content"]),
    "spawn_teammate": lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates": lambda **kw: TEAM.list_all(),
    "send_message": lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox": lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False),
    "broadcast": lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
}

# 工具列表 LLM看
TOOLS = [
    {"name": "bash",
     "description": "Run a shell command",
     "input_schema": {
         "type":"object",
         "properties": {"command": {"type": "string"}},
         "required": ["command"]}},
    {"name": "read_file",
     "description": "Read file contents",
     "input_schema": {
         "type":"object",
         "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
         "required": ["path"]}},
    {"name": "write_file",
     "description": "Write content to file",
     "input_schema": {
         "type":"object",
         "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
         "required": ["path", "content"]}},
    {"name": "edit_file",
     "description": "Replace exact text in file.",
     "input_schema": {
         "type":"object",
         "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
         "required": ["path", "old_text", "new_text"]}},
    {"name": "spawn_teammate",
     "description": "Spawn a persistent teammate that runs in its own thread.",
     "input_schema": {
         "type": "object",
         "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}},
         "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates",
     "description": "List all teammates with name, role, status.",
     "input_schema": {
         "type": "object",
         "properties": {}}},
    {"name": "send_message",
     "description": "Send a message to a teammate's inbox.",
     "input_schema": {
         "type": "object",
         "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}},
         "required": ["to", "content"]}},
    {"name": "read_inbox",
     "description": "Read and drain the lead's inbox.",
     "input_schema": {
         "type": "object",
         "properties": {}}},
    {"name": "broadcast",
     "description": "Send a message to all teammates.",
     "input_schema": {
         "type":"object",
         "properties": {"content": {"type":"string"}},
         "required": ["content"]}},
]

# 循环控制程序不断操作
def agent_loop(messages:list):  # 这里的输入是完整的对话历史记录/上下文 同时也包括用户输入
    while True:
        # 每次思考前先读取收件箱消息
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2, ensure_ascii=False)}</inbox>",
            })
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
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"> {block.name}:")
                print(output[:200])
                results.append({"type":"tool_result","tool_use_id":block.id,"content":output})
        # 把结果加入到上下文
        messages.append({"role":"user", "content":results})


if __name__ == "__main__":
    history = []
    while True:  # 实现用户和agent无限聊天
        try:
            # 真正的用户输入
            query = input("\033[36mv08>>\033[0m")
        except(EOFError, KeyboardInterrupt):  # 用户按ctrl+c可以退出
            break
        if query.strip().lower() in ("q", "exit", ""):  # 输入q/exit 可以退出
            break
        # 用/team可以看团队工作状态
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        # /inbox查看收件箱
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False))
            continue
        history.append({"role":"user", "content":query})
        agent_loop(history)
        # 取出并显示LLM最后返回的内容
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()