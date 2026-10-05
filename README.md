# HumanSlice

Desktop tool for heuristic slicing of human vocal materials for Vocaloid, UTAU, OpenUtau, and related voice-editing workflows.

HumanSlice 是一个面向人声素材整理的桌面切分工具，重点解决这条实际工作流：

`导入长音频 -> 自动生成候选切点 -> 人工微调 -> 快速试听 -> 导出片段与元数据`

它不追求音素级强对齐，而是把“切得更顺手、挑得更快、导出更方便”做成一个轻量可用的桌面应用。

## 功能特性

- 导入 `wav` / `mp3` / `flac` 音频
- 显示波形、频谱预览、整段 F0 预览
- F0 预览后台异步生成，导入较长素材时不会明显阻塞界面
- 基于短时能量、onset strength、spectral flux 与静音边界生成候选切点
- 对呼吸音碎切与中文词内弱边界做后处理合并
- 支持双击新增切点、拖拽切点、删除切点
- 支持撤销 / 重做、保存工程 / 打开工程
- 为每个片段分析 `onset / nucleus / tail`
- 为每个片段给出 `clarity_score`、`stability_score`、`recommended_role`
- 支持整段试听、片段试听、边界试听、相邻片段 crossfade 拼接试听
- 导出 `wav` 片段、`metadata.json`、`metadata.csv`、`oto.ini`、`utau_labels.csv`

## 适用场景

- 人声采样素材初步整理
- Vocaloid / UTAU / OpenUtau 相关切片准备
- 鬼畜 / 语音拼接 / 人声编辑前处理
- 需要先筛掉明显不稳、不清晰片段的离线工作流

## 环境要求

- Python 3.11+
- 建议使用虚拟环境
- GUI 基于 `PySide6`

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` 会以可编辑模式安装本项目（依赖声明在 `pyproject.toml`）。

## 启动

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

也可以用 `python -m humanslice`，或安装后直接运行 `humanslice`。

## 基本使用

1. 点击“导入音频”加载一段素材。
2. 波形会先显示出来，F0 预览会在后台补齐。
3. 点击“自动切分”生成候选切点。
4. 在波形区单击片段，或双击新增切点。
5. 拖动橙色切点线微调边界，必要时删除错误切点。
6. 通过片段试听、边界试听、拼接试听快速筛选可用素材。
7. 编辑 `alias` 和备注。
8. 点击“导出”输出片段与元数据，或先保存为 `project.json` 继续下次工作。

常用快捷键：

- `Ctrl+O` 导入音频
- `Ctrl+Shift+O` 打开工程
- `Ctrl+S` 保存工程
- `Ctrl+R` 自动切分
- `Ctrl+E` 导出
- `Ctrl+Z` 撤销
- `Ctrl+Shift+Z` / `Ctrl+Y` 重做
- `Space` 播放当前片段
- `Shift+Space` 播放整段
- `Alt+Left` / `Alt+Right` 切换片段
- `Delete` 删除选中切点
- `Ctrl+D` 进入添加切点模式（在波形或频谱上单击添加）
- `Escape` 取消添加切点模式 / 停止播放

## 配置

仓库提供了示例配置文件 [`config.example.json`](./config.example.json)。

如果需要覆盖默认参数，可复制为 `config.json` 后再启动程序。当前可调内容包括：

- 切点最小间隔、静音检测阈值与静音最短时长
- 呼吸音识别与弱边界合并阈值
- `onset / nucleus / tail` 分区参数
- F0 跟踪与预览降采样参数
- 试听 crossfade 时长与音量

## 导出内容

导出目录默认包含：

- `segments/`：逐片段导出的 `wav`
- `metadata.json`：完整结构化元数据
- `metadata.csv`：便于筛选与脚本处理的表格版本
- `oto.ini`：按启发式规则生成的 OTO 初值
- `utau_labels.csv`：表格化导出的 alias / offset / consonant / cutoff / preutterance / overlap

`oto.ini` 当前定位是“继续微调的起点”，不是成品级原音参数。

## 算法说明

### 候选切点

当前版本使用轻量启发式信号处理流程：

- `short-time energy` 用于识别能量抬升与低能区段
- `onset strength` 用于检测明显起音
- `spectral flux` 用于捕捉频谱突变
- 静音边界检测用于补充停顿前后的切分提示

这些线索先生成较宽松的候选切点，再做一次后处理：

- 合并呼吸音内部的碎切
- 合并中文词内较短、较弱的边界
- 尽量保留真实停顿和更强的词间边界

### 片段区域

每个片段会进一步划分为：

- `onset`：起音更明显的前段
- `nucleus`：通常更稳定、适合拉长或调音的主体区
- `tail`：更容易出现衰减和噪声的尾段

这一步仍然是启发式分析，不是音素级对齐。

### 评分

- `clarity_score`：综合起音强度、尾部残留能量、静音比例与时长合理性
- `stability_score`：综合 nucleus 区的 F0 稳定性、能量稳定性、voiced 概率与时长合理性
- `recommended_role`：给出 `onset` / `sustain` / `general` / `weak` 的经验性建议

### F0 预览

整段 F0 预览默认基于 `librosa.pyin`，并包含：

- `voiced_prob` 置信度过滤
- cents 域平滑
- 八度跳变惩罚
- 长音频自动降采样与点数控制

## 项目结构

```text
HumanSlice/
├─ main.py                  # 启动入口（等价于 python -m humanslice）
├─ pyproject.toml           # 包元数据、依赖、pytest 配置
├─ config.example.json
├─ humanslice/
│  ├─ __main__.py
│  ├─ app/                  # 主窗口与后台任务
│  ├─ analysis/             # 切点生成、区域分析、片段评分
│  ├─ audio/                # 音频加载、裁切、试听
│  ├─ models/               # 项目与片段数据结构
│  ├─ services/             # 工程保存、设置、导出、撤销重做、片段重建
│  └─ ui/                   # 波形 / 频谱 / F0 视图、片段详情面板、进度框
└─ tests/
```

目录职责：

- `humanslice/app/`：主窗口（只负责 UI 交互与状态编排）与 F0 后台任务
- `humanslice/analysis/`：切点生成、区域分析、片段评分等信号处理
- `humanslice/audio/`：音频加载、裁切、试听
- `humanslice/models/`：项目与片段数据结构
- `humanslice/services/`：工程保存、设置加载、导出、撤销重做，以及切点变化后的片段重建（`segment_service`，不依赖 Qt）
- `humanslice/ui/`：可复用的界面组件
- `tests/`：单元测试

## 测试

安装测试依赖：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

运行测试：

```powershell
pytest
```

当前测试覆盖了切点生成、呼吸音后处理、片段区域分析、评分、导出、撤销重做、工程保存加载、片段重建、波形视图基础行为，以及主窗口的切点编辑冒烟流程。

## 已知限制

- 当前版本会把整段音频读入内存，对超长素材不够节省内存。
- `mp3` 支持依赖本地解码后端，极端环境下仍可能解码失败。
- F0 跟踪是轻量离线启发式方案，不是专业级音高标注工具。
- 呼吸音识别和中文词内合并仍是规则法，不是语音学意义上的精确分段。
- `oto.ini` 是启发式初值，适合作为后续微调起点。
- `project.json` 保存的是工程状态，不嵌入原始音频数据。
