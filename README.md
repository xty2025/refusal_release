# safety 目录说明（按文件）

本目录当前主要项目是 `refusal_direction/`。下面按文件逐条说明用途，并明确区分：
- `纯LLM`：仅文本模型/文本指令流程
- `音频`：Qwen2-Audio 或 TTS 音频流程
- `通用`：两者共用

## 1) 顶层文件（`refusal_direction/`）

- `refusal_direction/.env` `[通用]`：本地环境变量（如 `HF_TOKEN`、`TOGETHER_API_KEY`）。
- `refusal_direction/.gitignore` `[通用]`：Git 忽略规则。
- `refusal_direction/LICENSE` `[通用]`：开源许可证。
- `refusal_direction/README.md` `[通用]`：原项目总览与主 pipeline 用法。
- `refusal_direction/STEERING_SCORE_EXPLAINED.md` `[通用]`：`steering_score/ablation/KL` 指标解释文档。
- `refusal_direction/requirements.txt` `[通用]`：依赖清单。
- `refusal_direction/setup.sh` `[通用]`：环境初始化脚本（Conda + PyTorch + token 配置）。

### 顶层实验脚本
- `refusal_direction/test_add_vec_output_txts.py` `[纯LLM]`：读取已选方向后做向量注入（文本输入），输出 txt 统计。
- `refusal_direction/test_audio_original_formula_persona.py` `[音频]`：按“原公式”选择的方向在 Qwen2-Audio 上做 persona 注入测试。
- `refusal_direction/test_add_audio_qwen_output_txts.py` `[音频]`：对音频模型做 baseline vs actadd，并输出多份 txt/json 统计。
- `refusal_direction/test_audio_all_layers_persona_tune.py` `[音频]`：在所有层统一注入向量，比较安全率变化。
- `refusal_direction/test_qwen_audio_safety.py` `[音频]`：Qwen2-Audio 的基础安全/误拒测试脚本。

## 2) 数据与数据脚本（`refusal_direction/dataset/`）

- `dataset/__init__.py` `[通用]`：包初始化。
- `dataset/load_dataset.py` `[纯LLM]`：加载 `splits/` 与 `processed/` 文本数据集。
- `dataset/generate_split_tts.py` `[音频]`：把 `splits/*.json` 文本批量 TTS 成 `splits_audio/`。
- `dataset/generate_datasets.ipynb` `[通用]`：数据处理/生成实验 notebook。

### 数据说明文档
- `dataset/DATASET_GUIDE.md` `[通用]`：数据集结构、来源与使用方式。
- `dataset/TTS_INTRO.md` `[音频]`：TTS 功能简介。
- `dataset/TTS_WORKFLOW.md` `[音频]`：TTS 操作步骤文档。
- `dataset/AUDIO_TRAINING_WORKFLOW.md` `[音频]`：音频训练流程文档。

### 结果/中间数据文件
- `dataset/test_results_qwen_audio.json` `[音频]`：Qwen2-Audio 测试结果（通用测试集）。
- `dataset/test_results_qwen_audio_harmful.json` `[音频]`：有害测试集结果。
- `dataset/test_results_qwen_audio_harmless.json` `[音频]`：无害测试集结果。
- `dataset/audio_original_formula_persona_test.json` `[音频]`：`test_audio_original_formula_persona.py` 的输出结果。
- `dataset/audio_original_formula_selected_direction.json` `[音频]`：选中的方向元信息（pos/layer/score）。

## 3) 主流程入口（`refusal_direction/pipeline/`）

- `pipeline/__init__.py` `[通用]`：包初始化。
- `pipeline/config.py` `[纯LLM]`：文本 pipeline 参数配置（train/val/test 数量、评测开关等）。
- `pipeline/run_pipeline.py` `[纯LLM]`：文本主流程入口：提方向 -> 选方向 -> 评估 -> loss。
- `pipeline/pipeline_audio.py` `[音频]`：音频主流程入口：音频样本匹配、提方向、方向评估、生成与统计。
- `pipeline/pipeline_audio_refusal_boost.py` `[音频]`：音频 refusal-boost 版本（按加权目标选“更拒绝有害”的方向）。
- `pipeline/train_qwen2_audio_from_splits.py` `[音频]`：基于 `splits_audio` 的 Qwen2-Audio 监督训练脚本。
- `pipeline/run_audio_refusal_boost_no_abla.sh` `[音频]`：一键跑训练+audio refusal boost+摘要导出的 shell 脚本。

### pipeline 文档
- `pipeline/Audio_related.md` `[音频]`：音频模型与阈值问题说明。
- `pipeline/PIPELINE_AUDIO_AUDIO_ONLY_RUN.md` `[音频]`：audio-only 运行记录与指标解释。

## 4) 模型适配层（`refusal_direction/pipeline/model_utils/`）

- `model_utils/__init__.py` `[通用]`：包初始化。
- `model_utils/model_base.py` `[通用]`：统一模型抽象基类（加载、tokenize、生成、模块访问）。
- `model_utils/model_factory.py` `[通用]`：按 `model_path` 分发到具体模型适配器。
- `model_utils/qwen_model.py` `[纯LLM]`：Qwen 文本模型适配。
- `model_utils/qwen_audio_model.py` `[音频]`：Qwen2-Audio 适配（`language_model` 结构、音频模型兼容）。
- `model_utils/llama2_model.py` `[纯LLM]`：Llama2 文本模型适配。
- `model_utils/llama3_model.py` `[纯LLM]`：Llama3 文本模型适配。
- `model_utils/gemma_model.py` `[纯LLM]`：Gemma 文本模型适配。
- `model_utils/yi_model.py` `[纯LLM]`：Yi 文本模型适配。

## 5) 核心子模块（`refusal_direction/pipeline/submodules/`）

- `submodules/generate_directions.py` `[通用]`：计算 harmful/harmless 均值激活差，生成候选方向。
- `submodules/select_direction.py` `[通用]`：对候选方向计算 refusal/steering/KL 并筛选最佳方向。
- `submodules/evaluate_jailbreak.py` `[通用]`：越狱评估（substring、llamaguard2、harmbench）。
- `submodules/evaluate_loss.py` `[通用]`：CE loss/perplexity 评估与 token inspection 工具。
- `submodules/Audio_qwen2_trainer.py` `[音频]`：Qwen2-Audio 最后一层 token/激活检查示例。

## 6) 工具层（`refusal_direction/pipeline/utils/`）

- `utils/__init__.py` `[通用]`：包初始化。
- `utils/hook_utils.py` `[通用]`：hook 上下文管理与方向消融/注入函数。
- `utils/utils.py` `[通用]`：矩阵对方向正交化的数学工具函数。
- `utils/token_get.md` `[通用]`：hook 获取 token/激活与干预流程详解。

## 7) 示例脚本（`refusal_direction/pipeline/examples/`）

- `examples/token_inspection_example.py` `[纯LLM]`：token inspection 的多种使用示例。
- `examples/test_qwen_audio.py` `[音频]`：Qwen2-Audio 兼容性与层级 token 检查示例。

## 8) 目录级别区分（快速看）

- **纯LLM主线**：`pipeline/run_pipeline.py` + `model_utils/{qwen,llama2,llama3,gemma,yi}_model.py` + `dataset/load_dataset.py`
- **音频主线**：`pipeline/pipeline_audio.py`、`pipeline/pipeline_audio_refusal_boost.py`、`pipeline/train_qwen2_audio_from_splits.py`、`dataset/generate_split_tts.py`、`model_utils/qwen_audio_model.py`
- **共用能力**：`submodules/*`（大部分）、`utils/*`、`evaluate_jailbreak.py`、`select_direction.py`

## 9) 说明边界

本文档聚焦“可开发/可运行文件”。以下目录一般是运行产物或环境文件，未逐文件展开：
- `.git/`
- `venv/`
- `dataset/raw/`, `dataset/processed/`, `dataset/splits/`, `dataset/splits_audio/`
- `dataset/output_*`
- `pipeline/runs/`
