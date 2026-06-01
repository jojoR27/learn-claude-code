# Learn-Claude-Code 学习重建项目

本项目是我对 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 2026.5.20之前版本的**个人学习复现项目**，
用于系统学习与实践 LLM 智能体（Agent）的核心架构，包括工具调用、上下文管理、技能系统、任务调度等关键模块。

---

## 项目说明
- **原项目参考**：https://github.com/shareAI-lab/learn-claude-code
- **本项目定位**：个人学习版重建，并非官方 Fork
- **目标**：从零实现一套完整的 Coding Agent，理解每一层逻辑并适配 Windows 环境

我在原项目基础上：
- 按自己的节奏调整了学习顺序
- 修复了 Windows 环境下的路径、编码、权限等问题
- 添加了详细中文注释与运行日志
- 重点实践了技能系统与上下文压缩模块

---

## 我的学习路线（已按个人节奏调整）

| 步骤      | 模块/主题                                 | 对应原项目           | 状态     |
|:--------|:--------------------------------------|:----------------|:-------|
| step 1  | 开发环境搭建 & 基础客户端配置                      | 环境              | ✅ 已完成  |
| step 2  | 基础工具循环（bash / read_file / write_file） | s01 + s02       | ✅ 已完成  |
| step 3  | Todo 管理与简单任务循环                        | s03 todo        | ✅ 已完成  |
| step 4  | 任务系统（Task System）                     | s07 task system | ✅ 已完成  |
| step 5  | 子代理（Sub-agent）与任务委派                   | s04 subagent    | ✅ 已完成  |
| step 6  | 外部技能系统（Skill Loader）                  | s05 skills      | ✅ 已完成  |
| step 7  | 三层上下文压缩（无限会话）                         | s06 compact     | ✅ 已完成  |
| step 8  | 后台任务与异步处理                             | s08 background  | ✅ 已完成  |
| step 9  | 模块 9                                  | s09             | ✅ 已完成  |
| step 10 | 模块 10                                 | s10             | ✅ 已完成  |
| step 11 | 模块 11                                 | s11             | ✅ 已完成  |
| step 12 | 模块 12                                 | s12             | ✅ 已完成  |
| step 13 | 整合                             | |  ⏳ 待完成 |

---

## 已实现核心模块

### ✅ V05 - 外部技能系统（s05 skills）
- 实现了 `SkillLoader`，可从外部 `*.md` 文件加载技能
- 支持 YAML Frontmatter 配置解析
- 解决了 Windows 下文件编码（GBK/UTF-8）与路径问题
- 模型可通过 `load_skill` 工具按需加载外挂知识

### ✅ V06 - 三层上下文压缩（s06 compact）
- **Layer 1：微压缩**：自动清理旧工具结果，仅保留最近 3 条
- **Layer 2：自动压缩**：Token 超限时自动总结对话
- **Layer 3：手动压缩**：模型可调用 `compact` 工具主动触发总结
- 所有对话会自动归档到 `.transcripts/` 目录，便于回溯

---

## 使用说明
- **具体参考原项目readme文件**
- **衍生路径**
-    V1-v2-v3-v4
-    v2-v5
-    v2-v6
-    v2-v7
-    v2 + 部分v7 - v8 - v9 - v10
-    v3-2 - v11
-    原版的full版出worktree(v11)没有融合进去之外，其他全融合；此项目full全融合