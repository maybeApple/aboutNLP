# Paper Knowledge Graph

一个面向论文场景的知识图谱项目。它从 `CSV/JSONL` 论文记录中读取标题与摘要，调用 LLM 抽取 `TASK / METHOD / DATASET / METRIC / RESULT`，将结果写入 Neo4j，并提供 Streamlit 检索界面。

<img width="1191" height="622" alt="image" src="https://github.com/user-attachments/assets/d5a081d2-cf6b-4c7c-b4e3-9d32532095f0" />


## 项目内容

- 论文数据导入：支持 `csv`、`jsonl`，也支持单文件 `docx` / `txt` 的兼容导入
- 论文知识抽取：围绕 `Paper` 节点构建固定 schema 的实体和关系
- 图谱写入与查询：Neo4j 约束、索引、实体检索、关系检索、上下文查询
- Web 界面：Streamlit 页面支持实体查询、关系查询、自然语言检索
- 语义检索：优先使用 `sentence-transformers`，缺失时自动回退到 TF-IDF
- 数据准备脚本：可从 OpenAlex 和 arXiv 抓取公开论文记录

## 技术栈

- Python 3.10-3.12
- Neo4j 5.x
- Streamlit
- OpenAI / DashScope / Anthropic
- `pandas`、`python-docx`、`tiktoken`
- `scikit-learn`，可选 `sentence-transformers`

## 图谱 Schema

节点类型：

- `Paper`
- `Task`
- `Method`
- `Dataset`
- `Metric`
- `Result`

关系类型：

- `Paper -[:STUDIES]-> Task`
- `Paper -[:PROPOSES]-> Method`
- `Paper -[:USES_DATASET]-> Dataset`
- `Paper -[:EVALUATED_BY]-> Metric`
- `Method -[:APPLIED_TO]-> Task`
- `Method -[:TESTED_ON]-> Dataset`
- `Method -[:ACHIEVES]-> Result`

## GitHub 建议保留的文件

建议公开仓库至少保留这些内容：

- 核心源码：`main.py`、`config.py`、`document_processor.py`、`embedding_manager.py`、`knowledge_graph_builder.py`、`llm_extractor.py`、`neo4j_manager.py`、`query_interface.py`
- 辅助脚本：`scripts/`
- 示例数据：`sample_papers.csv`、`data/papers_bootstrap_240.csv`
- 复现文件：`README.md`、`.gitignore`、`.env.example`、`requirements.txt`、`requirements-optional.txt`、`requirements-full.txt`、`environment.yml`、`docker-compose.yml`

不建议上传：

- `.env`
- `logs/`
- `runtime/`
- `__pycache__/`
- 本地工作记录：`WORKLOG.md`、`NEXT_SESSION.md`、`中文指导`
- 历史遗留文件：`lecture14.doc`

这些内容已经被写入 `.gitignore`，初始化 git 后执行 `git add .` 时会自动避开。

## 目录说明

| 文件 / 目录 | 作用 |
| --- | --- |
| `main.py` | CLI 入口，支持 `process`、`web`、`test` |
| `document_processor.py` | 论文记录读取、字段归一化、token 统计 |
| `llm_extractor.py` | LLM 抽取论文实体与关系 |
| `neo4j_manager.py` | Neo4j 连接、schema、写入与查询 |
| `knowledge_graph_builder.py` | 串联文档处理、抽取和入库 |
| `query_interface.py` | Streamlit 查询界面 |
| `scripts/fetch_paper_sources.py` | 从 OpenAlex / arXiv 抓取论文元数据 |
| `scripts/import_papers_only.py` | 只导入 `Paper` 节点，不触发 LLM 抽取 |
| `sample_papers.csv` | 小规模演示数据 |
| `data/papers_bootstrap_240.csv` | 240 条公开论文样例数据 |

## 环境复现

### 方式 1：venv + Docker Neo4j

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose up -d neo4j
```

如果你想启用密集向量检索，再执行：

```bash
python -m pip install -r requirements-optional.txt
```

### 方式 2：Conda

```bash
conda env create -f environment.yml
conda activate paper-kb
pip install -r requirements-optional.txt
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose up -d neo4j
```

### 方式 3：本地 Neo4j 或已打包运行时

如果本地已经安装 Neo4j，或者你手头仍然保留 `runtime/` 目录，可以直接使用：

```powershell
.\scripts\start_neo4j.ps1
```

这个脚本现在的行为是：

- 若存在 `runtime/jdk-21` 和 `runtime/neo4j`，优先启动本地打包 Neo4j
- 若不存在，则回退到 `docker compose up -d neo4j`

## 环境变量

先复制模板：

```bash
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
```

最少需要配置：

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- 至少一组 LLM 提供商密钥：
  - `OPENAI_API_KEY`
  - 或 `DASHSCOPE_API_KEY`
  - 或 `ANTHROPIC_API_KEY`

常见 DashScope 配置示例：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=please-change-this-password

DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DEFAULT_LLM_MODEL=qwen3-max
```

## 快速开始

### 1. 导入示例数据

```bash
python main.py process sample_papers.csv --provider dashscope --model qwen3-max --clear-database
```

### 2. 启动 Web 界面

```bash
python main.py web
```

或者在 Windows 下用辅助脚本：

```powershell
.\scripts\start_streamlit.ps1
```

### 3. 处理后直接打开页面

```bash
python main.py process sample_papers.csv --provider dashscope --model qwen3-max --web
```

### 4. 批量抓取公开论文

```bash
python scripts/fetch_paper_sources.py --count 240 --output data/papers_bootstrap_240.csv
```

### 5. 只导入论文元数据

```bash
python scripts/import_papers_only.py data/papers_bootstrap_240.csv --clear-database
```

### 6. 分批执行 LLM 抽取

```bash
python main.py process data/papers_bootstrap_240.csv --provider dashscope --model qwen3-max --start 0 --count 40
```

后续继续处理：

- `--start 40 --count 40`
- `--start 80 --count 40`
- `--start 120 --count 40`
- `--start 160 --count 40`
- `--start 200 --count 40`

## 输入格式

支持输入：

- `csv`
- `jsonl`
- `docx`
- `txt`

`csv/jsonl` 至少需要：

- `title`
- `abstract`

可选字段：

- `paper_id`
- `year`
- `authors`

兼容别名：

- `paper_title` -> `title`
- `summary` -> `abstract`
- `id` / `arxiv_id` -> `paper_id`
- `published_year` -> `year`
- `author` -> `authors`

当前不支持：

- `pdf`
- `doc`

## 运行机制

1. `main.py process` 读取输入文件
2. `document_processor.py` 归一化论文记录
3. `llm_extractor.py` 调用 LLM 抽取实体和关系
4. `neo4j_manager.py` 写入 `Paper`、实体节点和关系
5. `query_interface.py` 从 Neo4j 读取结果并在 Streamlit 展示

## 说明与限制

- 自然语言查询目前是“检索增强展示”，不会自动生成最终综合答案
- `Result` 节点仍偏自由文本，数据规模变大后可能比较噪声
- 未安装 `sentence-transformers` 时，语义检索会退化为 TF-IDF
- `main.py test` 是轻量自检；若未配置 API key，LLM 测试项会失败，但不影响项目结构本身
