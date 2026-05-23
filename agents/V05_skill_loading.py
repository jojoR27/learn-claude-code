'''V05版本的agent实现内容：
        1.skill.md: Agent的外挂知识包/技能包

        对应关系：bash---run_bash()
                read_file---run_read()
                write_file---run_write()
                edit_file---run_edit()
                load_skill---SkillLoader.get_content()'''



import os
import subprocess
from pathlib import Path
import re
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
SKILLS_DIR = Path(__file__).parent.parent / "skills"


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


SKILL_LOADER = SkillLoader(SKILLS_DIR)

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.
Skills available:
{SKILL_LOADER.get_descriptions()}"""


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error:Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
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
        output = (out + err).strip()
        return output[:5000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error:Timeout (120s)"
    except Exception as e:
        return f"Error:{e}"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines)-limit} more lines"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error:{e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
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


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_content"], kw["new_content"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}

TOOLS = [
    {"name": "bash",
     "description": "Run a shell command",
     "input_schema": {
         "type": "object",
         "properties": {"command": {"type": "string"}},
         "required": ["command"]}},
    {"name": "read_file",
     "description": "Read file contents",
     "input_schema": {
         "type": "object",
         "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
         "required": ["path"]}},
    {"name": "write_file",
     "description": "Write content to file",
     "input_schema": {
         "type": "object",
         "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
         "required": ["path", "content"]}},
    {"name": "edit_file",
     "description": "Replace exact text in file.",
     "input_schema": {
         "type": "object",
         "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
         "required": ["path", "old_text", "new_text"]}},
    {"name": "load_skill",
     "description": "Load specialized knowledge by name.",
     "input_schema": {
         "type": "object",
         "properties": {"name": {"type": "string", "description": "Skill name to load"}},
         "required": ["name"]}},
]


def agent_loop(messages: list):
    while True:
        print("\n[Agent] 正在请求模型思考...")

        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        print("[Agent] 模型返回回答！")
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            print("[Agent] 不再调用工具，结束循环！")
            return

        print(f"[Agent] 模型需要调用工具：{response.stop_reason}")
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[Agent] 正在执行工具：{block.name}")
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"> {block.name}: {output[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    print("=" * 60)
    print(" V05 Skill Agent 已启动！")
    print("=" * 60)

    while True:
        try:
            query = input("\n\033[36mv05>>\033[0m ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        print(f"\n[用户输入]：{query}")
        history.append({"role": "user", "content": query})
        agent_loop(history)

        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print("\n[Agent 最终回答]：")
                    print(block.text)
        print()