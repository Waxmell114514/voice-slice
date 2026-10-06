# HumanSlice

近乎全自动的人力音 MAD（人力 VOCALOID）制作工具。

```
素材库（某人的语音 / 视频） ──► 音素级切片 + 单元库 ──► UTAU 音源（OpenUtau 可直接加载）
目标歌曲（清唱或完整歌曲）  ──► 音高 / 歌词 / 音符     ──► OpenUtau 工程（USTX）+ MIDI
                         单元选择 + 声码器拼接 ──► 人力人声 wav（可与伴奏混音）
```

它不训练模型，也不是"AI 翻唱"：输出由素材里真实的音节片段拼成，只把音高、时长修到旋律上。
单元选择会优先挑音高接近、时长合适的片段，并偏好素材中原本连续的整词，以保留人力的味道。

目前只支持普通话。

## 环境要求

- Windows（开发与测试环境：Windows 11、Python 3.13、RTX 5070 Laptop 8 GB）
- Python 3.11+；NVIDIA 显卡（8 GB 显存够用，模型逐个加载）。也能用 CPU 运行（未充分测试，会很慢）
- [ffmpeg](https://ffmpeg.org/) 在 PATH 中：视频素材和 m4a / aac / wma 需要它，其余格式可不装
- 可选但推荐：[Miniconda](https://docs.conda.io/en/latest/miniconda.html)，用于安装 Montreal Forced Aligner（MFA）。
  没有 MFA 时自动退回 Qwen3-ForcedAligner，声母 / 韵母边界会粗一些
- 磁盘：模型约 7 GB，MFA 环境约 6 GB
- **项目路径不能含空格**（MFA 在含空格的路径下无法运行）

## 安装

```powershell
git clone https://github.com/Waxmell114514/voice-slice.git human-slice
cd human-slice
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu130   # RTX 50 系需要 CUDA 12.8+ 的版本
pip install -r requirements.txt

# 推荐：MFA 装在项目内的 data\envs\mfa（独立 conda 环境）
conda create -p .\data\envs\mfa -c conda-forge montreal-forced-aligner python=3.12
```

模型在首次用到时自动下载：Qwen3-ASR-1.7B、Qwen3-ForcedAligner-0.6B、RMVPE（Hugging Face），
PC-NSF-HiFiGAN、MFA 普通话模型（GitHub Releases），BS-Roformer 分离模型。
模型、MFA 环境和所有缓存都放在项目内的 `data\`（已加入 .gitignore），不会写到用户目录；
可用环境变量 `HUMANSLICE_HOME` 改到别处。

国内网络提示：

- Hugging Face 下载慢或卡住：设置 `$env:HF_ENDPOINT = "https://hf-mirror.com"`；下载停滞时再加 `$env:HF_HUB_DISABLE_XET = "1"`
- conda 下载中断：`conda create ... -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge --override-channels`
- pip：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

## 使用

### 准备材料

- **素材**：一个文件夹，放同一个人说话的录音或视频（wav / mp3 / flac / m4a / mp4 / mkv / flv 等，含子文件夹）。
  越干净、越长越好，能凑齐的字越全（开发时用的是约 110 分钟）。
  - 某个文件有现成字幕时，放一个同名的 `.txt` 或 `.lab`（如 `001.wav` 配 `001.txt`），会跳过语音识别，更准
  - 素材有背景音乐时打开"素材含背景音乐"（命令行 `--separate-material`），先做人声分离
- **歌曲**：清唱或带伴奏的完整歌曲。完整歌曲打开自动分离（命令行 `--separate`），
  会拆出人声来分析，成品再混回伴奏
- **歌词**（可选，强烈建议）：`.txt` 或 `.lrc`（时间标签会被忽略）。不给则自动识别，错误会多一些

### 图形界面

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

点主界面的 **"一键人力…"**，选择素材文件夹、歌曲、歌词和输出文件夹，点"开始制作"。
窗口里显示进度，完成后可以直接播放成品或打开输出文件夹。

### 命令行

```powershell
python -m humanslice make --material D:\素材\某人 --vocal D:\歌\歌曲.mp3 --separate --lyrics D:\歌\歌词.txt --name 某人 -o out\作品
```

`out\作品` 中会生成：

| 文件 | 说明 |
|---|---|
| `jinriki.wav` | 人力人声 |
| `jinriki_mix.wav` | 人力人声 + 原伴奏（`--separate` 时） |
| `jinriki.report.json` | 每个字用了素材里哪一段（来源文件、时间、变调量），方便排查 |
| `voicebank\<名字>\` | OpenUtau 中文 CVVC 音源（CV、`- CV`、VC、`V R` 别名），复制到 OpenUtau 的 Singers 文件夹即可 |
| `project.ustx` | OpenUtau 工程：音符 + 拼音歌词 + 原唱音高曲线（pitd），歌手已设为上面的音源，可手工精修 |
| `melody.mid` | 带拼音歌词的 MIDI |
| `score.json` | 分析出的乐谱（音符、歌词、音高曲线） |
| `bank\` | 单元库 |

第一次运行最慢，主要花在素材的识别和对齐上。做下一首歌时用 `--bank` 指定已有的单元库，
会跳过这一步；往素材文件夹里追加文件后再运行，只处理新增的部分：

```powershell
python -m humanslice make --material D:\素材\某人 --bank out\作品\bank --vocal 新歌.mp3 --separate -o out\新歌
```

常用调整参数（`make` 和 `render` 通用）：

- `--join-weight 3`：更偏好素材中原本连续的整词，人力感更强（默认 1）
- `--transpose N`：整体移调 N 个半音（默认自动按八度移到说话人音域，保持与伴奏同调）
- `--pitch-follow 0.5`：减弱对原唱颤音、滑音的跟随（1 = 完全跟随，0 = 平直音符）
- `--backend world`：改用 WORLD 声码器（不依赖非商用权重，音色更"电子"）
- `python -m humanslice --device cpu make ...`：强制用 CPU

更细的修改：把音源放进 OpenUtau，打开 `project.ustx`，改个别字的音符或歌词后在 OpenUtau 里渲染。

### 分步使用

```powershell
# 素材 -> 单元库（可多次追加素材）
python -m humanslice bank build D:\素材 -o bank [--separate]
# 单元库 -> OpenUtau 音源
python -m humanslice bank utau bank -o voicebank\某人 --name 某人
# 歌曲 -> 乐谱（给歌词最准；不给则自动识别）
python -m humanslice song analyze 清唱.wav --lyrics 歌词.txt -o score.json --ustx song.ustx --singer 某人 --midi song.mid
# 乐谱 + 单元库 -> 人力人声
python -m humanslice render bank score.json -o jinriki.wav
```

每个子命令都可以加 `-h` 查看全部参数。

### 手动切片

主界面本身仍是原来的手动切片工具：导入音频后自动生成候选切点，可拖动 / 新增 / 删除切点，
试听片段与拼接效果，编辑 alias，导出 wav 片段、`metadata.json/csv`、`oto.ini`。
常用快捷键：`Ctrl+O` 导入、`Ctrl+R` 自动切分、`Ctrl+E` 导出、`Ctrl+Z` / `Ctrl+Y` 撤销 / 重做、
`Space` 播放片段、`Ctrl+D` 进入添加切点模式（在波形上单击添加）、`Delete` 删除切点。
切分参数可复制 `config.example.json` 为 `config.json` 后修改。

## 工作原理与选型

调研结论：没有现成的端到端开源"自动人力"工具，但每个环节都有成熟方案，这里直接复用：

| 环节 | 方案 | 说明 |
|---|---|---|
| 分离 / 去 BGM | python-audio-separator（BS-Roformer） | 素材去背景音乐；完整歌曲拆成人声 + 伴奏 |
| 语音识别 | Qwen3-ASR-1.7B（transformers 原生） | 普通话识别准确率高，也能识别歌声 |
| 字级时间戳 | Qwen3-ForcedAligner-0.6B | 用于歌声对齐；素材侧作为 MFA 不可用时的后备 |
| 声母 / 韵母边界 | Montreal Forced Aligner 3.4 | 以带调拼音为 token，发音表由 MFA 词典自动推导 |
| 音高 | RMVPE | 歌声 F0 标准方案；八度错误修正 |
| 音符 | 按音节的 F0 分段（启发式） | 可替换为 GAME |
| 单元选择 | Viterbi（目标代价 + 拼接代价） | 音高差、拉伸倍数、质量、声调；素材内连续片段拼接代价为 0 |
| 合成 | PC-NSF-HiFiGAN（默认）/ WORLD | 对数 mel / 频谱参数域拼接，F0 用原唱曲线 |
| 音源 / 工程 | OpenUtau ZH CVVC（presamp.ini）/ USTX | 生成 oto.ini、character.yaml、pitd 曲线 |

渲染时每个字：辅音按原长放在拍点之前（先行发声），只拉伸元音中频谱最稳定的部分（主元音），
超过 2.5 倍时在稳定段内往返循环并平滑；复合元音的滑音和鼻音韵尾放在音符末尾；
相邻音节在参数域做 30 ms 交叉淡化。

## 当前效果与局限

开发集指标（GTSinger 男高音朗读约 110 分钟作素材，女中音演唱的《成都》《云烟成雨》作目标）：

- 素材对齐：音节起止平均误差约 21 ms（93% < 50 ms）；无字幕时识别的拼音错误率 2.3%
- 歌声分析（给定歌词）：音节起点 81% < 50 ms；音符逐帧音高命中 84–86%
- 选片：97% 的字找到同音节片段，变调中位数约 1 个半音
- 渲染：用 ASR 回听的音节错误率 7.4%（原唱 0.6%），音高中位误差 5 音分

已知局限：

- 只支持普通话
- 说话里的元音很短，长音只能循环延长，这是目前音质的主要瓶颈
- 素材里缺的音节暂时用韵母相同的字顶替，还没有"声母 + 韵母"跨字拼接
- 音符提取是启发式的，复杂旋律（快速跑音、大量装饰音）会出错

## 开发与评测

```powershell
pip install -r requirements-dev.txt
pytest

# 开发数据：GTSinger 子集（CC BY-NC-SA 4.0）下载到 devdata/
# （男高音的朗读录音作素材，女中音演唱的两首歌作目标）
python scripts/fetch_devdata.py --singer ZH-Tenor-1 --song-singer ZH-Alto-1 --song 成都 --song 云烟成雨
# 素材侧对齐精度（对照人工标注）
python scripts/eval_alignment.py
# 用开发语料建单元库，再准备两首目标歌曲
python -m humanslice bank build devdata/corpus -o out/bank_full
python scripts/gt_song.py devdata/songs/ZH-Alto-1_成都_Breathy -o out/chengdu
python scripts/gt_song.py devdata/songs/ZH-Alto-1_云烟成雨_Breathy -o out/yunyan
# 渲染可懂度（ASR 音节错误率）+ 音高误差
python scripts/bench.py out/bank_full
```

## 目录结构

```text
humanslice/
├─ cli.py            命令行入口（python -m humanslice ...）
├─ pipeline.py       一键流程（make 命令与 GUI 共用）
├─ corpus/           素材 -> 单元库：导入、切句、识别、MFA 对齐、特征、质检
├─ song/             目标人声 -> 乐谱：音高、歌词、对齐、音符
├─ render/           单元选择、时间规划、WORLD / NSF 合成
├─ export/           UTAU 音源、USTX、MIDI
├─ pitch/ text/      RMVPE 与 F0 工具；普通话 G2P
├─ common/           音频 I/O、TextGrid、模型下载、人声分离、数据目录
├─ ui/make_dialog.py "一键人力"对话框
├─ analysis/ app/ audio/ models/ services/ ui/   手动切片 GUI
scripts/             开发数据下载与评测脚本
tests/               单元测试
data/                模型、MFA 环境、缓存（自动生成，不入库）
```

## 许可与使用须知

- 本仓库代码为 MIT 许可。`humanslice/pitch/rmvpe.py` 的模型结构改编自
  [Applio](https://github.com/IAHispano/Applio)（MIT），源自 RVC-Project。
- 默认声码器 PC-NSF-HiFiGAN 的权重为 CC BY-NC-SA 4.0（仅限非商业用途）；`--backend world` 可完全避开。
  其余模型权重遵循各自发布方的许可。
- 开发数据 GTSinger 为 CC BY-NC-SA 4.0，不随仓库分发。
- 使用他人声音素材请自行确认授权与平台规则；OpenVPI 等上游项目明确反对未经同意合成他人声音。
