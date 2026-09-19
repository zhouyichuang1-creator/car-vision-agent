# PROJECT_CONTEXT — 项目上下文说明

> **给 AI 助手 / 新接手者的交接文档。读完本文件即可完整理解本项目。**
> 最后更新：2026-09-17

---

## 0. 一句话概括

一个**「看图识车 · 智能百科助手」**：用户上传车辆照片 → 自训练的 51 类图像分类模型识别车型 → DeepSeek 大模型通过 Function Calling 调用工具查资料 / 记历史 / 收藏 → FastAPI Web 界面呈现。

---

## 1. 项目背景与任务要求

课程：**视觉智能体综合实训**（4 周综合实训）。

任务书核心要求：

| 要求 | 说明 |
| --- | --- |
| 视觉模型 | 在教师下发的 **827 个车型类别**中**自主选择 50 类**训练一个图像分类模型 |
| unknown 类 | 模型需有第 51 个类别 `unknown`，**除选中的 50 类外都算 unknown**（开放集识别） |
| 评分指标 | **51 类 Macro-F1**（unknown 与每个已知类**等权**，不是准确率） |
| 智能体 | 接入大模型（本实现用 DeepSeek），需要 **≥3 个工具**、**≥1 个串联 3 个以上工具的任务**、**多轮上下文** |
| Web 应用 | 提供可交互的网页应用 |
| 提交物 | `classes.txt`（50 行）、`predictions.csv`、实训报告、源代码 |
| 禁止事项 | 不使用第三方在线图片识别 API；API Key 不得写入代码或报告 |

**评分口径提醒**：`Macro-F1` 对每个类一视同仁，所以**不能让 unknown 类别过强或过弱**——这是本项目的核心难点，详见第 7 节。

---

## 2. 最终成果（关键指标）

| 指标 | 数值 |
| --- | --- |
| **51 类 Macro-F1（验证集 1928 张）** | **0.9025** |
| Accuracy | 0.8849 |
| unknown 类 F1 | 0.8061 |
| 演示测试集（204 张）真实准确率 | 89.7% |
| 模型 | ResNet18 + 51 类分类头，11.21 M 参数 |
| 训练成本 | CPU，30 epoch，约 1 小时 45 分 |

模型演进（同一验证集、同口径）：

| 版本 | Macro-F1 | 说明 |
| --- | --- | --- |
| v1 | 0.7483 | 初版：15 epoch，unknown 1500 张（过采样） |
| **v2（当前）** | **0.8757** | 修正 unknown 至 500 张 + 30 epoch + MixUp + 余弦退火 |
| **v2 + TTA（最终提交口径）** | **0.9025** | 叠加 224px + 水平翻转 TTA |

---

## 3. 技术方案

### 3.1 视觉模型

- **Backbone**：ResNet18（torchvision `IMAGENET1K_V1` 预训练权重），迁移学习
- **Head**：`nn.Linear(512, 51)`
- **训练**：192×192 输入，batch 16，AdamW（lr 1e-3 → 余弦退火至 1e-5），CrossEntropyLoss，30 epoch
- **数据增强**：`RandomResizedCrop` + `RandomHorizontalFlip` + `ColorJitter` + **MixUp（α=0.2）**
- **推理**：224×224 + **水平翻转 TTA**（原图与翻转图两路概率平均）
- **模型选择**：按验证集 51 类 Macro-F1 选 best checkpoint

### 3.2 智能体

- **大模型**：DeepSeek，接口 `https://api.deepseek.com/chat/completions`，模型名 `deepseek-v4-flash`
- **调用方式**：Function Calling（tools + tool_choice=auto），自实现多轮工具循环
- **API Key**：只从环境变量 `DEEPSEEK_API_KEY` 读取（`python-dotenv` 加载本地 `.env`），**代码中无任何明文 Key**

**5 个工具**（定义在 `src/tools/`）：

| 工具 | 作用 |
| --- | --- |
| `classify_car(image_path)` | 调用本项目的视觉模型识别车型，返回 top1 + top-k + 置信度 |
| `query_car_info(car_id)` | 查车型资料（中文名 / 英文名 / 车身类型），**仅限 50 个已知类** |
| `add_to_history(car_id, image_id, confidence)` | 写入浏览历史（SQLite） |
| `mark_favorite(history_id)` | 标记收藏 |
| `list_history(limit)` | 查询历史记录 |

**串联任务示例**（任务书要求 ≥3 工具串联）：
> 用户上传图片 + "这是什么车？介绍一下，再存进历史"
> → `classify_car` → `query_car_info` → `add_to_history` → 汇总回答

### 3.3 Web 应用

- **后端**：FastAPI（`src/web/app.py`），端点：`/`、`/api/upload`、`/api/chat`
- **前端**：原生 HTML/CSS/JS + marked.js 渲染 Markdown，三栏布局（会话信息 / 对话 / 工具调用日志）
- **图片传递机制**：前端上传后拿到绝对路径，作为文本 hint 拼进用户消息，由 agent 按需调用 `classify_car`

---

## 4. 目录结构

```
├── README.md                     # 项目简介与快速开始
├── PROJECT_CONTEXT.md            # 本文件（AI 交接文档）
├── .gitignore                    # 上传规则（分区注释）
├── .env.example                  # 环境变量模板（提交）
├── requirements.txt              # 依赖清单
├── classes.txt                   # ★ 交付物：50 个车型 ID，按升序、UTF-8 无 BOM
├── predictions.csv               # ★ 交付物：204 条预测（表头 image_id,predicted_label,confidence）
│
├── src/
│   ├── config.py                 # 全局配置；API Key 读取；拒识阈值
│   ├── data_prep.py              # 选类 / unknown 采样（子命令 inspect|make-classes|make-unknown|write-final）
│   ├── train.py                  # 训练入口（支持 --mixup-alpha / --cosine / --lr-min）
│   ├── evaluate.py               # 验证集评估（支持 --tta / --img-size / --temperature）
│   ├── inference.py              # 单图推理（含 TTA、拒识回退、温度校准）
│   ├── batch_predict.py          # 批量推理 → predictions.csv
│   ├── agent.py                  # DeepSeek Function Calling 主循环 + Session
│   ├── report_to_docx.py         # 把 report/*.md 转成 docx（自动嵌图）
│   ├── tools/                    # Agent 工具（每个文件暴露 TOOL_DEFINITIONS / TOOL_FUNCTIONS / run）
│   ├── utils/                    # dataset.py（Dataset）/ model.py（建模+存取）/ metrics.py（Macro-F1 等）
│   └── web/                      # FastAPI 应用 + templates + static
│
└── report/
    ├── 实训报告.md / .docx        # ★ 交付物：完整实训报告（含全部实验数据）
    └── 提交前自检.md
```

**未提交到仓库的目录**（见第 9 节）：`数据集/`、`checkpoints/`、`data_local/`、`logs/`、`.workbuddy/`、`.env`

---

## 5. 如何运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（复制模板后填入真实 Key，.env 不会被提交）
cp .env.example .env
#    然后编辑 .env，设置 DEEPSEEK_API_KEY=sk-xxxxx

# 3. 启动 Web（默认 http://127.0.0.1:8000）
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000

# 4. 单图推理（命令行）
python -m src.inference single --image 路径/到/图片.jpg

# 5. 批量推理
python -m src.batch_predict --test-dir 测试图目录 --out predictions.csv --tta flip

# 6. 评估（需先有模型权重）
python -m src.evaluate --img-size 224 --tta flip --make-confusion-png
```

**Windows 环境注意**：使用 Anaconda 的 Python 时需设置 `KMP_DUPLICATE_LIB_OK=TRUE`，否则会因 libiomp5md.dll 重复加载而崩溃。

---

## 6. 如何复现训练

> 需要教师下发的数据集（`数据集/train/<4位ID>/`、`数据集/val/<4位ID>/`、`数据集/cls_info/class_info.json`），本仓库不含数据。

```bash
# ① 选 50 类（生成 classes.txt）
python -m src.data_prep make-classes --seed 42 --out classes.txt

# ② 采样 unknown 训练/验证集（注意：会同时写入 train 和 val）
python -m src.data_prep make-unknown --per-src-train 10 --max-train 500 \
       --per-src-val 12 --max-val 400 --seed 42

# ③ 训练（最终配置）
python -m src.train --epochs 30 --batch-size 16 --img-size 192 \
       --mixup-alpha 0.2 --cosine --lr-min 1e-5 --seed 42

# ④ 评估（最终口径）
python -m src.evaluate --img-size 224 --tta flip --make-confusion-png
```

**关于模型权重**：`best.pt`（44.8 MB）不随仓库分发。若需要：
- 方案一：按上面步骤自行训练（CPU 约 1 小时 45 分）
- 方案二：从本项目的 GitHub Release 下载 `best.pt`，放到 `checkpoints/best.pt`

---

## 7. 开发历程与关键决策

### 7.1 时间线

| 阶段 | 内容 | 结果 |
| --- | --- | --- |
| 数据探索 | 发现数据集实际是 **827 类、ID 范围 0000–0998 且不连续**（任务书示例写的是 0–826，与实际不符） | 按实际 ID 处理 |
| 选类 | 从 827 类中随机抽 50 类（seed=42，过滤样本数 < 60 的类） | `classes.txt` |
| v1 训练 | ResNet18 / 15 epoch / 192px / unknown 1500 张 | Macro-F1 0.7483 |
| 推理优化 | 试验 192px→224px、flip TTA、5-crop TTA、温度校准 | 224px+flip 最优（0.7723）；5-crop 与温度校准**均被否决** |
| 问题诊断 | 用户反馈"识别已知车型时置信度只有 50%，识别 unknown 时反而 80–90%" | **定位根因：unknown 被过采样 12.5×** |
| 事后补救 | 在验证集标定拒识回退阈值 τ | 增益仅 +0.0008，**属噪声级，无法根治** |
| **v2 重训** | unknown 降到 500 + 30 epoch + MixUp + 余弦退火 | **Macro-F1 0.8757，叠加 TTA 达 0.9025** |
| 交付 | 重跑 predictions.csv、重画 4 张图、改写报告、端到端回归 | 全部对齐 v2 |

### 7.2 关键决策（含被否决的方案）

| 决策 | 结论 | 依据 |
| --- | --- | --- |
| unknown 训练样本量 | **1500 → 500**（占比 24.5% → 9.8%） | 见下方 7.3，这是本项目最大的一次改进 |
| 推理尺寸 | **224px**（训练用 192px 以节省时间） | v2 上仍 +0.0229 |
| TTA 策略 | **只用水平翻转**；5-crop 被否决 | 车辆主体居中占满画面，四角裁剪切掉判别性区域，v1 上掉 0.0131 |
| 温度校准 | **不启用**（T=1.0） | 温度缩放不改变 argmax，对 Macro-F1 零影响，只会改变置信度数值标度 |
| 拒识回退阈值 τ | **0.90（等价于不触发）** | v2 标定发现 τ≥0.40 收益完全持平，说明模型自身已校准良好 |
| 学习率策略 | 固定 1e-3 → **余弦退火 1e-3→1e-5** | v1 后期震荡（ep11–14 在 0.67~0.73 波动） |

### 7.3 核心问题与解法（本项目的技术亮点）

**问题**：v1 模型识别已知车型时置信度偏低（约 50%），而识别 unknown 时反而很高（80–90%）。

**诊断过程**：

1. 统计训练集构成，发现 **unknown 类被过采样 12.5 倍**：
   - 训练集 6115 张 = 50 个已知类 4615 张（平均 92.3 张/类）+ unknown **1500 张**
   - unknown 占比 **24.5%**，而均匀分布下应为 **2.0%**（1/51）
2. 由混淆矩阵验证后果：**11.85% 的已知类图片被误判为 unknown**；unknown 精确率仅 0.6351
3. **决定性证据**：仅把 unknown 概率归零后重新归一化，已知类准确率就从 75.85% → **80.89%（+5.04%）**——说明有 77 张图模型本来能认出，却被 unknown 的虚高概率"抢"走了 top1

**根治**：把 unknown 降到 500 张（与其他类同量级）并重训。效果：

| 指标 | v1 | v2 |
| --- | --- | --- |
| 已知图被误判为 unknown | 11.85% | **6.87%** |
| 单类最高误判率 | 42.3% | **0.6%** |
| unknown 精确率 / 召回率 | 0.6351 / 0.7875 | **0.7646 / 0.8525**（同涨） |
| Macro-F1 | 0.7483 | **0.8757** |

**方法论提示**：这个案例说明——当模型的"拒识"行为异常时，**先查数据分布，再查训练轮数**。事后用决策规则打补丁救不回来（实测增益 +0.0008，噪声级）。

### 7.4 一个需要注意的副作用

v2 引入 MixUp 后，模型输出整体更保守：**预测正确样本的平均置信度从 0.8925 降到 0.8199**。但同期**预测错误样本的置信度从 0.5744 降到 0.4413**——即模型"自信地答错"的情况显著减少，置信度的**区分度**反而更好，低置信度成为可靠的"需要人工确认"信号。这是有意的权衡，不是缺陷。

---

## 8. 踩坑记录（开发中真实遇到并修复的问题）

| 问题 | 原因 | 修复 |
| --- | --- | --- |
| `history.json` 写盘崩溃 | `per_class_f1` / `confusion_matrix` 是 numpy 数组，不可 JSON 序列化 | 保存前转 list |
| 推理结果与训练评估不一致 | `save_checkpoint` 把 `img_size` 写死为 224 | 参数化；推理优先读 `extra.infer_img_size` |
| 工具调用失败（`Path` 收到 dict） | 工具注册时存了业务函数而非执行入口 | 改为注册模块级 `run(tool_call, name)` |
| `query_car_info` 能查到 50 类外的车型 | 未限制查询范围 | 增加 `classes.txt` 白名单校验 |
| Web 上传按钮点击无反应 | 对 `hidden`（display:none）的 file input 调用 `.click()` 被浏览器忽略 | 改用 `<label for>` 转发点击，input 改为 `.visually-hidden`；另加整页拖拽上传 |
| 重启 Web 报端口占用 | 旧进程未退出 | 先 `Get-NetTCPConnection -LocalPort 8000 \| Stop-Process` |
| 后台长任务被 60 秒杀掉 | 前台超时 | 长任务必须后台运行并重定向日志 |
| 重采样 unknown 差点破坏验证集 | 现成的 `make-unknown` 会**同时重写 train 和 val** | 另写脚本只动训练集，并用文件清单 md5 锁定校验验证集 |
| Python 源码 SyntaxError 导致服务起不来 | 双引号字符串内嵌 ASCII 直引号 | 中文引号放在 docstring 或用「」 |
| 长任务日志长时间空白 | stdout 重定向到文件时 Python 块缓冲 | 打印行加 `flush=True` |

---

## 9. GitHub 上传说明

### 9.1 会上传（30 个文件，全部为文本，体积 < 1 MB）

- 源代码：`src/` 全部
- 交付物：`classes.txt`、`predictions.csv`、`report/实训报告.md` + `.docx`
- 工程文件：`README.md`、`PROJECT_CONTEXT.md`、`requirements.txt`、`.env.example`、`.gitignore`

### 9.2 不上传（及原因）

| 不上传 | 原因 | 如何获取 |
| --- | --- | --- |
| `.env` | **含真实 API Key** | 从 `.env.example` 复制后自行填写 |
| `数据集/`、`数据集.zip` | 教师下发，版权不归本项目；10 万+ 张图约 6.6 GB，且 GitHub 单文件上限 100 MB | 使用自有数据集 |
| `视觉智能体综合实训：学生任务书.pdf`、`*.ipynb` | 教师提供的课程材料 | — |
| `checkpoints/`（`best.pt` 44.8 MB） | 二进制大文件会让 git 历史急剧膨胀 | 自行训练，或从 GitHub Release 下载 |
| `data_local/` | 本地 SQLite 收藏库、Web 上传的图片 | 运行时自动生成 |
| `logs/`、`.workbuddy/` | 训练日志与 AI 协作笔记 | — |

### 9.3 首次上传命令

```bash
# 1) 确认清单（应为 30~31 个文件、约 0.6 MB）
git status

# 2) 配置署名（换成你自己的 GitHub 用户名与邮箱；建议用 noreply 邮箱）
git config user.name  "<你的 GitHub 用户名>"
git config user.email "<你的 noreply 邮箱>"

# 3) 拆成 7 个功能 commit（见 9.5），最后一次 commit 后：
git branch -M main
git remote add origin https://github.com/<用户名>/<仓库名>.git
```

> 在 GitHub 网页新建仓库时，**不要勾选** "Add a README" / "Add .gitignore"，避免与本地冲突。

### 9.4 国内网络：git 走代理（不留残留配置）

Git **不读** Windows 系统代理，必须单独告诉它。Clash Verge 的混合端口默认 `127.0.0.1:7897`。

**推荐做法 —— 只在 push 那一行临时挂代理，不写任何持久配置：**

```bash
git -c http.proxy=socks5://127.0.0.1:7897 push -u origin main
```

这样**不需要**执行 `git config`，也就不存在"以后不用节点了要改回来"的问题。

若嫌每次要敲太长，可临时写进本仓库配置（只影响本项目，不影响其他仓库）：

```bash
git config --local http.proxy  socks5://127.0.0.1:7897
git config --local https.proxy socks5://127.0.0.1:7897
# 不用时一键清除：
git config --local --unset http.proxy; git config --local --unset https.proxy
```

**三个必须知道的点：**
1. 只用 `socks5://`，**不要用 `http://`**——Git Bash 的 curl 走 HTTP 代理连 HTTPS 目标会触发 Windows schannel 的 TLS 重协商死循环，表现为卡住后失败。
2. Clash 必须**保持运行**；退出代理软件后 git 立即断网。
3. Clash 里那个「系统代理」开关是给**浏览器**用的（注册 GitHub、建仓库、生成 Token 都在网页做），git 用不到它。

### 9.5 关于中文项目路径

**不需要修改，保持 `D:\智能系统综合实训\` 即可。** 理由：

1. `src/config.py` 里 `PROJECT_ROOT` 及 `数据集/`、`checkpoints/`、`data_local/` 全是拼出来的绝对路径，改目录名会让 Web 与 Agent 全部挂掉。
2. git 对中文路径支持完好，已实测：
   - `git hash-object "report/实训报告.docx"` → `899b4530…`（正常读取）
   - 执行 `git config core.quotepath false` 后，`git status` 显示 `report/实训报告.docx` 而非 `\346\225\260…` 转义码
3. GitHub 网页端、clone、CI 都能正确处理 UTF-8 中文路径。
4. 唯一要注意的是**别用 GBK 编码的旧版命令行工具**操作仓库（本项目统一用 Git Bash，无此问题）。

> 已设置的 `core.quotepath false` 是全局的，属于**纯显示优化**，无需还原。

### 9.6 手动操作清单（需要用户本人完成）

Git 侧的配置已全部就绪，下面是**只有账号持有人才能做**的部分。按 0→4 顺序做，做完第 4 步回来交给 AI 做 commit 拆分，最后执行第 5 步推送。

#### 第 0 步 · 打开 Clash 的「系统代理」开关

这一步是给**浏览器**用的（git 命令自己挂代理，见 9.4）。

1. 打开 Clash Verge → 左侧「设置」
2. 找到「系统代理」开关 → 打开（开关变蓝/变亮）
3. **验证**：浏览器打开 `https://github.com`，能正常显示页面即成功

> 若浏览器仍打不开：确认 Clash 左侧「代理」页选中的节点是测速正常的，且「代理模式」为「规则」或「全局」。

#### 第 1 步 · 注册 / 登录 GitHub

**没有账号：**
1. 浏览器打开 `https://github.com/signup`
2. 依次填：邮箱 → 密码（≥8 位，含数字和小写字母）→ 用户名（英文/数字/连字符，例如 `zhangsan-car`）
3. 去邮箱收 6 位验证码，填回页面
4. 可能弹出人机验证（拼图/旋转图形），按提示做完
5. 注册完成后进入首页

**已有账号：** 直接打开 `https://github.com/login` 登录。

**建议顺手做（可选但加分）：** 开启两步验证 —— 右上头像 → Settings → Password and authentication → Two-factor authentication → 用手机 Authenticator App 扫码开启。

> 记下你的**用户名**，第 3 步要提交给 AI。

#### 第 2 步 · 新建一个空仓库

1. 打开 `https://github.com/new`（或右上角 **+** → New repository）
2. **Repository name** 填 `car-vision-agent`（英文、无空格；可自取）
3. **Description** 可填：`51 类车型识别 + DeepSeek 智能体 + FastAPI Web 应用`
4. 选 **Public**（简历展示需要别人能看；先选 Private 也行，之后可在 Settings 里改回）
5. **⚠️ 关键：下面三个初始化选项一个都不要勾**
   - ❌ Add a README file
   - ❌ Add .gitignore
   - ❌ Choose a license
   > 勾了任何一个，GitHub 会在远端先生成一个 commit。本地已有自己的提交历史，两边没有共同祖先，push 会被拒绝（报 `non-fast-forward`）。
6. 点 **Create repository**
7. 创建成功后页面顶部会给出仓库地址，复制 **HTTPS** 那一条：
   `https://github.com/<用户名>/car-vision-agent.git`

#### 第 3 步 · 把三条信息交给 AI

1. **仓库地址**：`https://github.com/<用户名>/<仓库名>.git`
2. **GitHub 用户名**
3. **署名邮箱**：建议用 GitHub 隐私邮箱 —— Settings → Emails → 勾选 `Keep my email addresses private`，然后把页面上给出的 `<数字>+<用户名>@users.noreply.github.com` 复制过来（避免真实邮箱被公开爬取）

> AI 拿到这三条后会：设置 `user.name` / `user.email`、把 31 个文件拆成 7 个功能 commit、配置 `origin`。

#### 第 4 步 · 准备推送凭据（两条路任选其一）

本仓库已把凭据助手固定为 **Git Credential Manager 2.7.3**（`git config --local credential.helper`），所以第一次推送**不会**再弹那个令人困惑的「选择凭据助手」窗口。

**路 A（推荐）· 浏览器授权，一次搞定，之后永久免登录**

push 时会弹出一个 GitHub 登录窗口 → 点 **Sign in with your browser** → 浏览器里点绿色 **Authorize** 按钮 → 回到终端，推送自动继续。

前提：第 0 步的 Clash 系统代理是开着的（浏览器要能访问 github.com）。

**路 B · 用 Personal Access Token（PAT）**

1. 浏览器打开 `https://github.com/settings/tokens`
2. **Generate new token** → 选 **Generate new token (classic)**
3. **Note** 随便写，如 `local-push`；**Expiration** 选 90 天
4. 权限**只勾 `repo`** 这一项（它包含公有/私有仓库的读写与推送）
5. 拉到底点 **Generate token**
6. **⚠️ 令牌只在这一次显示**，立刻点复制并保存好（形如 `ghp_xxxxxxxx`）
7. 使用时二选一：
   - 在 push 弹窗里选 **Token** 方式，粘贴进去
   - 或用带令牌的地址推一次（令牌会进入命令行历史，仅在本机使用可接受，**不要截图外发**）：
     ```bash
     git -c http.proxy=socks5://127.0.0.1:7897 push https://<用户名>:<PAT>@github.com/<用户名>/<仓库名>.git main
     ```

> 出现过 `Authentication failed` 时，回报到 GitHub Settings → Applications 里撤销旧令牌，重新生成一个。

#### 第 5 步 · 最后执行推送

```bash
# 让 Git Credential Manager 能连上 GitHub（.NET 用它自己的 HTTP 栈，
# 走 http:// 形式的代理没有 schannel 那个 bug，可以放心用）
export HTTPS_PROXY=http://127.0.0.1:7897

# git 自身的传输用 socks5（原因见 9.4），-c 的优先级高于环境变量
git -c http.proxy=socks5://127.0.0.1:7897 push -u origin main
```

看到 `branch 'main' set up to track 'origin/main'` 即成功。刷新浏览器里的仓库页面，应能看到 31 个文件与 7 条提交记录。

#### 常见报错对照

| 报错关键词 | 原因 | 解决 |
| --- | --- | --- |
| `Could not resolve host: github.com` | Clash 没运行，或命令漏了 `-c http.proxy=…` | 打开 Clash；给命令补上 `-c http.proxy=socks5://127.0.0.1:7897` |
| `schannel` / `SSL_ERROR_SYSCALL` / 卡住几十秒后失败 | 代理写成了 `http://` | 改成 `socks5://127.0.0.1:7897` |
| `Authentication failed` / `403` | PAT 过期，或没勾 `repo` | 重新生成令牌并勾选 `repo` |
| `non-fast-forward` / `Updates were rejected` | 建仓库时勾了 README/gitignore/license | 删掉该仓库重建空仓库，或用 `git push --force` 覆盖远端 |
| `Please tell me who you are` | `user.name` / `user.email` 没设 | 执行 9.3 第 2 步 |
| `Support for password authentication was removed` | 把账号密码当密码填了 | GitHub 已禁用密码推送，必须用路 A 或路 B 的令牌 |

## 10. 当前状态与已知问题

### 已完成
- ✅ 视觉模型（51 类，Macro-F1 0.9025）
- ✅ DeepSeek 智能体（5 工具，含 3 工具串联任务、多轮上下文）
- ✅ FastAPI Web 应用（上传 / 对话 / 工具日志）
- ✅ 全部交付物（`classes.txt`、`predictions.csv`、报告 docx）
- ✅ 端到端回归验证（单图推理、Agent 链路、Web 全流程）

### 待办 / 已知问题
1. **报告 7.1 节有一处旧数据**：Web 实测置信度写的还是 v1 时代的 99.26%，建议用新模型重新实测后替换。
2. **报告 8.1 个人贡献**需填写（当前为 `______` 占位），docx 中同样留空。
3. **报告 4.6 节**建议补 2–3 张典型错误样本的图片说明。
4. **`predictions.csv` 为演示用**：当前基于 `data_local/test_set/`（51 类 × 4 张）生成；若教师提供正式测试集，需用 `src/batch_predict.py` 重新生成。
5. **后续优化方向**（按性价比，详见报告 8.3）：换更强 backbone（ResNet50 / ConvNeXt，预期 +0.03~0.05，CPU 4–6 小时）。

### 可回退点（本地）
- `checkpoints/v1_15ep/`：v1 模型的全部产物（用于对比）
- `数据集/train/_unknown_v1_1500/`：v1 的 1500 张 unknown 训练样本
