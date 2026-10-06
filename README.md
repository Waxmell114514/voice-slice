# HumanSlice

近乎全自动的人力音 MAD（人力 VOCALOID）制作工具。

```
素材库（某人的语音 / 视频） ──► 音素级切片 + 单元库 ──► UTAU 音源（OpenUtau 可直接加载）
目标歌曲（清唱或完整歌曲）  ──► 音高 / 歌词 / 音符     ──► OpenUtau 工程（USTX）+ MIDI
                         单元选择 + 声码器拼接 ──► 人力人声 wav（可与伴奏混音）
```

它不训练模型，也不是"AI 翻唱"：输出由素材里真实的音节片段拼成，只把音高、时长修到旋律上。
单元选择会优先挑音高接近、时长合适的片段，并偏好素材中原本连续的整词，以保留人力的味道。

## 快速开始

```powershell
# 1. 环境（Python 3.13 + NVIDIA 显卡；RTX 50 系需要 CUDA 12.8+ 的 PyTorch）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 2. 推荐：安装 Montreal Forced Aligner（音素级对齐，独立 conda 环境）
conda create -p .\data\envs\mfa -c conda-forge montreal-forced-aligner python=3.12

# 3. 一条命令：素材文件夹 + 歌曲 -> 人力人声 + 音源 + 工程
python -m humanslice make --material D:\素材\某人 --vocal 歌曲.mp3 --separate --lyrics 歌词.txt -o out\作品
```

`out\作品` 中会生成：

| 文件 | 说明 |
|---|---|
| `jinriki.wav` / `jinriki_mix.wav` | 人力人声 / 与原伴奏混音（`--separate` 时） |
| `jinriki.report.json` | 每个字用了素材里哪一段（来源文件、时间、变调量） |
| `voicebank/` | OpenUtau 中文 CVVC 音源（CV、`- CV`、VC、`V R` 别名） |
| `project.ustx` | OpenUtau 工程：音符 + 拼音歌词 + 原唱音高曲线（pitd），可手工精修 |
| `melody.mid` | 带拼音歌词的 MIDI |
| `bank/` | 单元库（可复用：再做别的歌时用 `--bank` 指定） |

首次运行会自动下载模型（约 8 GB）：Qwen3-ASR-1.7B、Qwen3-ForcedAligner-0.6B、RMVPE、
PC-NSF-HiFiGAN、BS-Roformer 分离模型、MFA 普通话模型。模型、MFA 环境和缓存全部放在项目内的
`data\` 文件夹（已加入 .gitignore；可用环境变量 `HUMANSLICE_HOME` 改到别处）。
**项目路径不能含空格**：MFA 在含空格的路径下无法运行。

## 分步使用

```powershell
# 素材 -> 单元库（可多次追加素材；同名 .lab/.txt 文件会作为现成转写，跳过识别）
python -m humanslice bank build D:\素材 -o bank [--separate]
# 单元库 -> OpenUtau 音源（把文件夹放进 OpenUtau 的 Singers 目录）
python -m humanslice bank utau bank -o voicebank\某人 --name 某人
# 歌曲 -> 乐谱（给歌词最准；不给则自动识别）
python -m humanslice song analyze 清唱.wav --lyrics 歌词.txt -o score.json --ustx song.ustx --singer 某人 --midi song.mid
# 乐谱 + 单元库 -> 人力人声
python -m humanslice render bank score.json -o jinriki.wav
```

常用渲染参数：

- `--transpose N`：整体移调（默认自动按八度移到说话人音域，保持与伴奏同调）
- `--join-weight`：越大越偏好素材中原本连续的整词（人力感更强）
- `--pitch-follow`：1 = 完全跟随原唱音高（含颤音、滑音），0 = 平直音符
- `--backend world`：改用 WORLD 声码器（不依赖非商用权重，音质略差）

不带参数运行 `python -m humanslice`（或 `python main.py`）打开原有的手动切片 GUI。

## 工作原理与选型

调研结论：没有现成的端到端开源"自动人力"工具，但每个环节都有成熟方案，这里直接复用：

| 环节 | 方案 | 说明 |
|---|---|---|
| 分离 / 去 BGM | python-audio-separator（BS-Roformer） | 素材去背景音乐；完整歌曲拆成人声 + 伴奏 |
| 语音识别 | Qwen3-ASR-1.7B（transformers 原生） | 普通话 CER 明显优于 Whisper，也能识别歌声 |
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

## 开发与评测

```powershell
pip install -r requirements-dev.txt
pytest
python scripts/fetch_devdata.py          # 下载 GTSinger 子集（CC BY-NC-SA）到 devdata/
python scripts/eval_alignment.py         # 语音对齐精度（对照人工标注）
python scripts/gt_song.py devdata/songs/ZH-Alto-1_成都_Breathy -o out/chengdu
python scripts/bench.py out/bank_full    # 渲染可懂度（ASR 音节错误率）+ 音高误差
```

开发集上的当前指标（GTSinger 男高音朗读 110 分钟作素材，女中音两首歌作目标）：

- 语音对齐：音节起止平均误差约 21 ms（93% < 50 ms）
- 歌声对齐（给定歌词）：音节起点 81% < 50 ms；音符逐帧音高命中 84%
- 渲染：ASR 音节错误率 7.4%（原唱 0.6%），音高中位误差 5 音分

## 目录结构

```text
humanslice/
├─ cli.py            命令行入口（python -m humanslice ...）
├─ corpus/           素材 -> 单元库：导入、切句、识别、MFA 对齐、特征、质检
├─ song/             目标人声 -> 乐谱：音高、歌词、对齐、音符
├─ render/           单元选择、时间规划、WORLD / NSF 合成
├─ export/           UTAU 音源、USTX、MIDI
├─ pitch/ text/      RMVPE 与 F0 工具；普通话 G2P
├─ common/           音频 I/O、TextGrid、模型下载、人声分离、数据目录
├─ analysis/ app/ audio/ models/ services/ ui/   原手动切片 GUI
scripts/             开发数据下载与评测脚本
tests/               单元测试
```

## 许可与使用须知

- 本仓库代码为 MIT 许可。
- 默认声码器 PC-NSF-HiFiGAN 的权重为 CC BY-NC-SA 4.0（仅限非商业用途）；`--backend world` 可完全避开。
- 开发数据 GTSinger 为 CC BY-NC-SA 4.0，不随仓库分发。
- 使用他人声音素材请自行确认授权与平台规则；OpenVPI 等上游项目明确反对未经同意合成他人声音。
