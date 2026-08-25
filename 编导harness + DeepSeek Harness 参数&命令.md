### DeepSeek Harness 配置&安装【必须】

**以下命令均在 WSL ubuntu 中验证；如果已经完成 Deepseek Harness SDK 安装，以下 DSH 安装步骤可省略【建议安装在dsharness_wd目录下】：**

```bash
# wsl 安装Ubuntu
wsl --install -d Ubuntu
# wsl 基础环境
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
  git \
  curl \
  wget \
  ca-certificates \
  build-essential \
  python3 \
  python3-venv \
  python3-pip

# git clone 步骤在仅使用 DSH 作测试时可省略，但需要安装 dsh-sdk
# 在 dsharness_wd 目录下，安装 deepseek harness
git clone https://github.com/deepseek-ai/deepseek-harness.git
# 下载速度慢时可选择添加 gh-proxy 前缀
git clone https://gh-proxy.com/https://github.com/deepseek-ai/deepseek-harness.git

# 在 dsharness_wd 目录下，新建虚拟环境(以 .venv 为例)
python3 -m venv .venv
source .venv/bin/activate

# linux 使用本地源码安装 Openharness（可选）
python -m pip install -e ./OpenHarness

# linux 安装 DSH_SDK
# 使用 DeepSeek Harness 时，以下命令需在当前 `writer_harness` 虚拟环境中执行一次。SDK 必须安装在启动 `writer_excute.py` 的同一个 Python 环境中：
python -m pip install --upgrade pip
python -m pip install --upgrade deepseek-harness-sdk

# 验证 DSH 配置
python -c "from deepseek_harness import DeepSeekHarness; print('SDK import: OK')"
python -c "from deepseek_harness_runtime import bundled_runtime_path; print(bundled_runtime_path())"
```



### 环境变量参数设置【重要】

基础环境配置【避免 director 执行出现错误】

```bash
cd /mnt/d/myDevelopKit/Python/workspace/dsharness_wd
source 你的python环境目录/bin/activate
export PROJECT_ROOT="dsharness_wd 文件夹绝对路径"
```

**【以下六项API相关参数执行时必须配置】**编导模式需要配置 Writer，Actor 配置由 OpenHarness 和 DeepSeek Harness 共用；Deepseek 模式暂不支持通过Director harness 影响执行过程，暂不配置 Director：

```bash
# linux
export WRITER_MODEL=''
export WRITER_BASE_URL=''
export WRITER_API_KEY=''

export ACTOR_MODEL=''
export ACTOR_BASE_URL=''
export ACTOR_API_KEY=''
```

OpenHarness 模式下，`OH_BIN` 默认使用 `oh`，单轮命令会自动使用项目内 `OpenHarness/src`，因此通常无需手动设置：

```bash
export ACTOR_API_FORMAT='openai'
```



### 执行命令【重要】

#### 基于 OpenHarness 真实执行（本次测试不需要）

```bash
python ./writer_excute.py \
  --query "请列出当前 OpenHarness 项目中与工具定义相关的文件，并说明哪些工具通常为只读。" \
  --mode writer_harness \
  --writer-model "$WRITER_MODEL" \
  --writer-base-url "$WRITER_BASE_URL" \
  --writer-api-key "$WRITER_API_KEY" \
  --actor-model "$ACTOR_MODEL" \
  --actor-base-url "$ACTOR_BASE_URL" \
  --actor-api-key "$ACTOR_API_KEY" \
  --actor-api-format "$ACTOR_API_FORMAT" \
  --oh-real-run \
  --execute-output-format stream-json \
  --json
```

`--actor-backend openharness`、`--oh-bin` 和 `--openharness-src` 使用默认值：分别为 `openharness`、`oh` 和项目内 `OpenHarness/src`。省略 `--oh-real-run` 时会执行 dry-run，不会真实调用 OpenHarness。

#### 基于 DeepSeek Harness 的 编导harness 执行【本次测试目标】

DSH 替代 OpenHarness 作为 Actor，不需要 `oh`、`--oh-real-run`、`ACTOR_API_FORMAT`，也不需要传入 Provider、运行目录或 Session 目录。两种 Actor 共用 `ACTOR_MODEL`、`ACTOR_BASE_URL` 和 `ACTOR_API_KEY`：

```bash
# Linux 环境
python ./writer_excute.py \
  --query "巴威和白海豚谁造成的危害更大" \
  --mode writer_harness \
  --writer-model "$WRITER_MODEL" \
  --writer-base-url "$WRITER_BASE_URL" \
  --writer-api-key "$WRITER_API_KEY" \
  --actor-backend deepseek-harness \
  --actor-model "$ACTOR_MODEL" \
  --actor-base-url "$ACTOR_BASE_URL" \
  --actor-api-key "$ACTOR_API_KEY" \
  --execute-output-format stream-json \
  --json
```

如需保存json格式文件，可选 `--json | tee dsh-result.json`

```bash	
python ./writer_excute.py   --query "请列出当前 deepseek harness项目中与工具定义相关的文件，并说明哪些工具通常只涉及读取操作。"   --mode writer_harness   --writer-model "$WRITER_MODEL"   --writer-base-url "$WRITER_BASE_URL"   --writer-api-key "$WRITER_API_KEY"   --actor-backend deepseek-harness   --actor-model "$ACTOR_MODEL"   --actor-base-url "$ACTOR_BASE_URL"   --actor-api-key "$ACTOR_API_KEY"   --json | tee dsh-result.json
```



OpenHarness 流式输出【本次测试不需要】

```bash
python ./writer_execute_stream.py \
  --query "请列出当前 DeepSeek Harness SDK 中与插件定义相关的文件，并说明哪些插件通常为只读。" \
  --mode writer_harness \
  --writer-model "$WRITER_MODEL" \
  --writer-base-url "$WRITER_BASE_URL" \
  --writer-api-key "$WRITER_API_KEY" \
  --actor-backend openharness \
  --oh-real-run \
  --actor-model "$ACTOR_MODEL" \
  --actor-base-url "$ACTOR_BASE_URL" \
  --actor-api-key "$ACTOR_API_KEY"
```
