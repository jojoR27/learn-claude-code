'''V04版本的agent实现内容：
        1.subagent:"大任务拆小, 每个小任务干净的上下文" -- Subagent 用独立 messages[], 不污染主对话, 守护模型的思维清晰度。

        对应关系：bash---run_bash()
                read_file---run_read()
                write_file---run_write()
                edit_file---run_edit()
                task_create---TaskManager.create()
                task_update---TaskManager.update()
                task_list--- TaskManager.list_all()
                task_get---TaskManager.get()
                task---run_subagent() # 单独的派生子代理工具 映射关系没写在工具映射表中直接写死在agent_loop中了'''



import os          # os 库就是 Python 和「操作系统」对话的工具，例如读取环境变量
import subprocess  # 让 Python 执行系统命令（cmd /bash 命令）
from pathlib import Path  # 用于后面限制文件操作的
import json   # 用来读写task.json文件

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
TASKS_DIR = WORKDIR / ".tasks"

# 父agent身份描述
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use task management tools to plan and track work.
Use the task tool to spawn subagent and handle complex subtasks."""
# subagent_prompt
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."

# task相关操作封装成一个类
class TaskManager:
    # 初始化：创建/索引json文件路径/ID
    def __init__(self, tasks_dir:Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    # 给task标序号
    def _max_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    # 读取json文件内容
    def _load(self, task_id: int) -> dict:
        path = self.dir/f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        # 读取并把json转成字典
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir/f"task_{task['id']}.json"
        # 字典转为json格式
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))

    def create(self, subject: str, description: str) -> str:
        task={
            "id": self._next_id,  # 任务编号
            "subject": subject,   # 任务标题
            "description": description,  # 任务详情
            "status": "pending",  # 状态
            "blockedBy": [],      # 依赖任务/父任务
            "owner": "",          # 执行者
        }
        self._save(task)
        self._next_id += 1
        # json 格式给LLM看
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        # 加载单个任务 json格式
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, remove_blocked_by: list = None) -> str:
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)
            if add_blocked_by:
                task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
            if remove_blocked_by:
                task["blockedBy"] = [x for x in task["blockedBy"] if x not in remove_blocked_by]
            self._save(task)
            return json.dumps(task, indent=2, ensure_ascii=False)

    # 清空依赖
    def _clear_dependency(self, completed_id: int):
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    # 列给LLM看所以任务
    def list_all(self) -> str:
        tasks = []
        files = sorted(
            self.dir.glob("task_*.json"),
            key=lambda f: int(f.stem.split("_")[1])
        )
        for f in files:
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks found"
        lines = []
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[X]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        return "\n".join(lines)

# 创建全局任务管理器实例
TASKS = TaskManager(TASKS_DIR)

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
                           timeout=120,
                           encoding='utf-8',
                           errors='replace'
                           )
        out = r.stdout if r.stdout else ""
        err = r.stderr if r.stderr else ""
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

# 对应工具task 派生子进程
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}] # 干净的上下文
    print(f"  [subagent] 开始执行任务: {prompt[:50]}...")
    for i in range(30):
        print(f"  [subagent] 第 {i + 1} 轮思考...")
        # 子Agent的单次输出
        response = client.messages.create(
            model = MODEL,
            system = SUBAGENT_SYSTEM,
            messages = sub_messages,
            tools = CHILD_TOOLS,
            max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)[:50000]})
        sub_messages.append({"role": "user", "content": results})
    return "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"

# 工具映射表
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_content"], kw["new_content"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"),kw.get("addBlockedBy"), kw.get("removeBlockedBy")),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
}

# subagent 工具列表
CHILD_TOOLS = [
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
]

# 工具列表 LLM看
TOOLS = CHILD_TOOLS + [
    {"name": "task_create",
     "description": "Create a new task.",
     "input_schema": {
         "type":"object",
         "properties": {"subject": {"type": "string"}, "description": {"type": "string"}},
         "required": ["subject"]}},
    {"name": "task_update",
     "description": "Update a task's status or dependencies.",
     "input_schema": {
         "type":"object",
         "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "removeBlockedBy": {"type": "array", "items": {"type": "integer"}}},
         "required": ["task_id"]}},
    {"name": "task_list",
     "description": "List all tasks with status summary.",
     "input_schema": {
         "type":"object",
         "properties": {}}},
    {"name": "task_get",
     "description": "Get full details of a task by ID.",
     "input_schema": {
         "type":"object",
         "properties": {"task_id": {"type": "integer"}},
         "required": ["task_id"]}},
    {"name": "task",  # 这个task工具和run_subagent()的映射关系没写在映射表中，直接在agent_loop中写死了
     "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
     "input_schema": {
         "type":"object",
         "properties": {"prompt": {"type": "string"}, "description": {"type": "string", "description": "Short description of the task"}},
         "required": ["prompt"]}},
]

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
                # 派生子agent单独对应
                if block.name == "task":
                    desc = block.input.get("description", "subtask")
                    prompt = block.input.get("prompt", "")
                    print(f"> task ({desc}): {prompt[:80]}")
                    output = run_subagent(prompt)
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"    {str(output)[:200]}")
                results.append({"type":"tool_result","tool_use_id":block.id,"content":str(output)})
        # 把结果加入到上下文
        messages.append({"role":"user", "content":results})


if __name__ == "__main__":
    history = []
    while True:  # 实现用户和agent无限聊天
        try:
            # 真正的用户输入
            query = input("\033[36mv04>>\033[0m")
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