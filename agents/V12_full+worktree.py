'''V12_full+worktree实现内容：
    （原项目的full版本是除了s12worktree其他所以功能的集合）
    本项目将所以的功能进行了融合，包括worktree
'''

import json
import os
import re
import subprocess
from pathlib import Path
from queue import Queue
import threading
import time
import uuid
from typing import Optional

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
client = Anthropic(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    default_headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
)
MODEL = os.getenv("MODEL_ID")


WORKDIR = Path.cwd()
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPTS_DIR = WORKDIR / ".transcripts"
TOKEN_THRESHOLD = 100000
POLL_INTERVAL = 5   # 轮询间隔
IDLE_TIMEOUT = 60
VALID_MSG_TYPES = {
    "message", "broadcast", "shutdown_request",
    "shutdown_response", "plan_approval_response"
}
# 压缩相关
THRESHOLD = 50000
KEEP_RECENT = 3
PRESERVE_RESULT_TOOLS = {"read_file"}

# v10
# 用于存放所有线程的请求
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()  # 因为所有线程都在操作上面的字典，所以要加锁
# 任务领取锁
_claim_lock = threading.Lock()

# 基础工具
def safe_path(p:str) -> Path:
    # 把用户输入的路径，拼接成当前工作目录下的绝对路径
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    # 路径安全返回拼接好的路径
    return path

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

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        # 文件不存在则创建
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {fp}"
    except Exception as e:
        return f"Error:{e}"

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

# ------------------------------------v11----------------------------------
# worktree日志总线
class EventBus:
    def __init__(self, event_log_path: Path):
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    # 写日志
    def emit(self,
             event: str,
             task: Optional[dict] = None,
             worktree: Optional[dict] = None,
             error: Optional[str] = None, ):
        payload = {
            "event": event,
            "ts": time.time(),
            "task": task or {},
            "worktree": worktree or {},
        }
        if error:
            payload["error"] = error
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    # 读最近的日志
    def list_recent(self,limit: int = 20) -> str:
        n = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        recent = lines[-n:]
        items = []
        for line in recent:
            try:
                items.append(json.loads(line))
            except Exception:
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, indent=2)

# 任务管理
class TaskManager:
    # 初始化：创建/索引json文件路径/ID
    def __init__(self, tasks_dir:Path):
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    # 给task标序号
    def _max_id(self) -> int:
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                pass
        return max(ids) if ids else 0

    def _path(self, task_id: int) -> Path:
        return self.dir / f"task_{task_id}.json"

    # 读取json文件内容
    def _load(self, task_id: int) -> dict:
        path = self.dir/f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        # 读取并把json转成字典
        return json.loads(path.read_text())

    def _save(self, task: dict):
        self._path(task["id"]).write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str) -> str:
        task = {
            "id": self._next_id,  # 任务编号
            "subject": subject,   # 任务标题
            "description": description,  # 任务详情
            "status": "pending",  # 状态
            "owner": "",          # 执行者
            "worktree": "",
            "blockedBy": [],  # 依赖任务/父任务 在这个版本中没用到
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save(task)
        self._next_id += 1
        # json 格式给LLM看
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        # 加载单个任务 json格式
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def exists(self, task_id: int) -> bool:
        return self._path(task_id).exists()

    def update(self, task_id: int, status: str = None, owner: str = None) -> str:
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if owner is not None:
               task["owner"] = owner
            task["updated_at"] = time.time()
            self._save(task)
            return json.dumps(task, indent=2, ensure_ascii=False)

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        task = self._load(task_id)
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        if task["status"] == "pending":
            task["status"] = "in_progress"
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def unbind_worktree(self, task_id: int) -> str:
        task = self._load(task_id)
        task["worktree"] = ""
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

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
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[X]"}.get(t["status"], "[?]")
            owner = f" owner: {t['owner']}" if t.get("owner") else ""
            wt = f" wt={t['worktree']}" if t.get("worktree") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{wt}")
        return "\n".join(lines)

# 工作树 实现task之间的隔离
class WorktreeManager:
    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus):
        self.repo_root = repo_root
        self.tasks = tasks
        self.events = events
        self.dir = repo_root / ".worktrees"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir/"index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2))
        self.git_available = self._is_git_repo()

    def _is_git_repo(self) -> bool:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> str:
        if not self.git_available:
            raise RuntimeError("Not in a git repository. worktree tools require git.")
        r = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode !=  0:
            msg = (r.stdout + r.stderr).strip()
            raise RuntimeError(msg or f"git {' '.join(args)} failed")
        return (r.stdout + r.stderr).strip() or "(no output)"

    def _load_index(self) -> dict:
        return json.loads(self.index_path.read_text())

    def _save_index(self, data: dict):
        self.index_path.write_text(json.dumps(data, indent=2))

    def _find(self, name:str) -> Optional[dict]:
        idx = self._load_index()
        for wt in idx.get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None

    def _validate_name(self, name:str):
        if not re.fullmatch(r"^[a-zA-Z0-9_]+$", name or ""):
            raise ValueError(f"Invalid name: {name}")

    def create(self, name: str, task_id: int = None, base_ref: str = "HEAD") -> str:
        self._validate_name(name)
        if self._find(name):
            raise ValueError(f"Name {name} already exists")
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id}not found")

        path = self.dir / name
        branch = f"wt{name}"
        self.events.emit(
            "worktree.create.before",
            task={"id": task_id} if task_id is not None else{},
            worktree = {"name": name, "base_ref": base_ref},
        )
        try:
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])

            entry = {
                "name": name,
                "path": str(path),
                "branch": branch,
                "task_id": task_id,
                "status": "active",
                "created_at": time.time(),
            }

            idx = self._load_index()
            idx["worktrees"].append(entry)
            self._save_index(idx)

            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)

            self.events.emit(
                "worktree.create.after",
                task={"id": task_id} if task_id is not None else {},
                worktree = {
                    "name": name,
                    "path":str(path),
                    "branch": branch,
                    "status": "active",
                    },
            )
            return json.dumps(entry, indent=2)
        except Exception as e:
            self.events.emit(
                "worktree.create.failed",
                task={"id": task_id} if task_id is not None else {},
                worktree={"name": name, "base_ref": base_ref},
                error=str(e),
            )
            raise

    def list_all(self) -> str:
        idx = self._load_index()
        wts = idx.get("worktrees", [])
        if not wts:
            return "No worktrees in index"
        lines = []
        for wt in wts:
            suffix = f" task={wt['task_id']}" if wt.get("task_id") else ""
            lines.append(
                f"[{wt.get('status', 'unknown')}] {wt['name']} -> "
                f"{wt['path']} ({wt.get('branch', '-')}){suffix}"
            )
        return "\n".join(lines)

    def status(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = (r.stdout + r.stderr).strip()
        return text or "Clean worktree"

    def run(self, name: str, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"

        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"

        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            out = (r.stdout + r.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (300s)"

    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        self.events.emit(
            "worktree.remove.before",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={"name": name, "path": wt.get("path")},
        )
        try:
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(wt["path"])
            self._run_git(args)

            if complete_task and wt.get("task_id") is not None:
                task_id = wt["task_id"]
                before = json.loads(self.tasks.get(task_id))
                self.tasks.update(task_id, status="completed")
                self.tasks.unbind_worktree(task_id)
                self.events.emit(
                    "task.completed",
                    task={
                        "id": task_id,
                        "subject": before.get("subject", ""),
                        "status": "completed",
                    },
                    worktree={"name": name},
                )

            idx = self._load_index()
            for item in idx.get("worktrees", []):
                if item.get("name") == name:
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)

            self.events.emit(
                "worktree.remove.after",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path"), "status": "removed"},
            )
            return f"Removed worktree '{name}'"
        except Exception as e:
            self.events.emit(
                "worktree.remove.failed",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path")},
                error=str(e),
            )
            raise

    def keep(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        idx = self._load_index()
        kept = None
        for item in idx.get("worktrees", []):
            if item.get("name") == name:
                item["status"] = "kept"
                item["kept_at"] = time.time()
                kept = item
        self._save_index(idx)

        self.events.emit(
            "worktree.keep",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={
                "name": name,
                "path": wt.get("path"),
                "status": "kept",
            },
        )
        return json.dumps(kept, indent=2) if kept else f"Error: Unknown worktree '{name}'"

# --------------------------------v03------------------------------------
# todo列表，LLM在要程序执行操作前生成，不再想到什么做什么
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i+1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[×]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for item in self.items if item["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

# --------------------------------v05-----------------------------------
# 加载外挂工具包
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        print("[SkillLoader] 开始加载技能...")
        self._load_all()
        print(f"[SkillLoader] 加载完成！可用技能：{list(self.skills.keys())}")

    def _load_all(self):
        if not self.skills_dir.exists():
            print("[SkillLoader] 警告：skills 文件夹不存在！")
            return
        print(f"[SkillLoader] 找到技能目录：{self.skills_dir}")
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            print(f"[SkillLoader] 发现技能文件：{f}")
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill: '{name}'. Available: {','.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"

# --------------------------------v06--------------------------------
# 上下文压缩
def estimate_tokens(messages: list) -> int:
    return len(str(messages)) // 4

def micro_compact(messages: list) -> list:
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
            if len(tool_results) <= KEEP_RECENT:
                return messages
            tool_name_map = {}
            for msg in messages:
                if msg["role"] == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if hasattr(block, "type") and block.type == "tool_use":
                                tool_name_map[block.id] = block.name
            to_clear = tool_results[:-KEEP_RECENT]
            for _, _, result in to_clear:
                if not isinstance(result.get("content"), str) or len(result["content"]) <= 100:
                    continue
                tool_id = result.get("tool_use_id", "")
                tool_name = tool_name_map.get(tool_id, "unknown")
                if tool_name in PRESERVE_RESULT_TOOLS:
                    continue
                result["content"] = f"[Previous: used {tool_name}]"
            return messages

def auto_compact(messages: list, focus: str = "") -> list:
    # Save full transcript to disk
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPTS_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # Ask LLM to summarize
    conversation_text = json.dumps(messages, default=str)[-80000:]
    focus_instruction = ""
    if focus:
        focus_instruction = f" Pay special attention to preserving details about: {focus}."
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details."
            f"{focus_instruction}\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = next((block.text for block in response.content if hasattr(block, "text")), "")
    if not summary:
        summary = "No summary generated."
    # Replace all messages with compressed summary
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
    ]

# --------------------------------------v07----------------------------------------
# 后台任务管理类 实现并行操作
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str):
        try:
            r = subprocess.run(
                command, shell =True, cwd = WORKDIR,
                capture_output = True,text = True, timeout = 300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "result": (output or "(no output)")[:500],
            })

    def check(self, task_id:str = None) -> str:
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"Error: Unknown task {task_id}"
            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"
        lines = []
        for tid, t in self.tasks.items():
            lines.append(f"{tid}: [{t['status']}] {t['command'][:60]}")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notification(self) -> list:
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs


# ---------------------------------------v10------------------------------------
# 消息总线类，每个agent一个jsonl文件为收件箱
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid types are {VALID_MSG_TYPES}"
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

# 扫描是否有未分配任务
def scan_unclaimed_tasks() -> list:
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")):
            unclaimed.append(task)
        return unclaimed

# 对应工具claim_task
def claim_task(task_id: int, owner: str) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"{task_id}.jsonl"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        if task.get("owner"):
            existing_owner = task.get("owner") or "someone else"
            return f"Error: Task {task_id} has already been claimed by {existing_owner}"
        if task.get("status") != "pending":
            status = task.get("status")
            return f"Error: Task {task_id} cannot be claimed because its status is '{status}'"
        if task.get("blockedBy"):
            return f"Error: Task {task_id} is blocked by other task(s) and cannot be claimed yet"
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    return f"Claimed task #{task_id} for {owner}"

# 身份重建上下文太长防止忘记身份 子agent从空闲 → 领任务 → 身份重建 → 继续干活
def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }

# 团队管理类，子agent团队
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

    # 子agent自己进行状态更新
    def _set_status(self, name: str, status: str):
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

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
        team_name = self.config["team_name"]
        sys_prompt = f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. Use idle tool when you have no more work. You will auto-claim new tasks."
        # 每个agent有独立的上下文
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        while True:
            # 循环执行任务
            for _ in range(50):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})

                try:
                    response = client.messages.create(
                        model=MODEL,
                        system=sys_prompt,
                        messages=messages,
                        tools=tools,
                        max_tokens=8000,
                    )
                except Exception:
                    self._set_status(name, "idle")
                    return

                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break

                results = []
                idle_requested = False  # 标记是否调用了idle
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name =="idle":
                            idle_requested = True
                            output = "Entering idle phase. Will poll for new tasks."
                        else:
                            output = self._exec(name, block.name, block.input)
                        print(f"  [{name}] {block.name}: {str(output)[:120]}")
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(output),
                        })
                messages.append({"role": "user", "content": results})

                if idle_requested:
                    break

            # 进入空闲轮询
            self._set_status(name, "idle")
            resume = False
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
            for _ in range(polls):
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break

                # 扫描领任务
                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    task = unclaimed[0]
                    result = claim_task(task["task_id"], name)
                    if result.startswith("Error:"):
                        continue
                    task_prompt = f"<auto-claimed>Task #{task['id']}: {task['subject']}\n. {task.get('description', '')}</auto-claimed>"
                    if len(messages) <= 3:
                        messages.append(0, make_identity_block(name, role, team_name))
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing"})
                    messages.append({"role": "user", "content": task_prompt})
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                    resume = True
                    break

                if not resume:
                    # 没认领到任务
                    self._set_status(name, "shutdown")
                    return
                self._set_status(name, "working")

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
        # 关闭响应工具
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = args["approve"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
            BUS.send(sender, "lead", "", "shutdown_response", {"request_id": req_id, "approve": approve})
            return f"Shutdown {'approved' if approve else 'rejected'}"
        # 提交计划审计工具
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)
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
            {"name": "shutdown_response",
             "description": "Respond to a shutdown request. Approve to shut down, reject to keep working.",
             "input_schema": {
                 "type": "object",
                 "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}},
                 "required": ["request_id", "approve"]}},
            {"name": "plan_approval",
             "description": "Submit a plan for lead approval. Provide plan text.",
             "input_schema": {
                 "type": "object",
                 "properties": {"plan": {"type": "string"}},
                 "required": ["plan"]}},
            {"name": "idle",
             "description": "Signal that you have no more work. Enters idle polling phase.",
             "input_schema": {
                 "type": "object",
                 "properties": {}}},
            {"name": "claim_task",
             "description": "Claim a task from the task board by ID.",
             "input_schema": {
                 "type": "object",
                 "properties": {"task_id": {"type": "integer"}},
                 "required": ["task_id"]}},
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

# 对应工具shutdown_request
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request_id {req_id} sent to '{teammate}' (status: pending)"

# 对应工具plan_approve_response
def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send(
        "lead",  req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"

# 对应工具shutdown_response
def check_shutdown_status(request_id: str) -> str:
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))

# 全局实例
TODO = TodoManager()
SKILLS = SkillLoader(SKILLS_DIR)
TASK_MGR = TaskManager()
BG = BackgroundManager()
BUS = MessageBus()
TEAM = TeammateManager(BUS, TASK_MGR)

# prompt
SYSTEM =


