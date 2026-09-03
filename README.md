# CD-FND: 跨域假新闻检测多智能体系统(科研代码)

本项目是把「Coordinator → Specialist → Judge → Optimizer」多智能体假新闻检测
流程落地为可训练、可复现、可做论文实验的代码基线。结构与设计决策遵循
`LOCAL_PROJECT_AUDIT.md`(本地内部文档,已 gitignore)。

> 设计原则:**科研用途,代码追求简单、确定、可复现,不做花哨技巧。**

## 一、系统闭环(两条运行路径)

```
训练路径:
  训练域样本
    → Coordinator(两阶段: 路由特征 → 固定 Specialist 目录选择)
    → 选中 Specialist 逐个分析(specialist_report_v1)
    → 聚合器(确定性合并为 analysis_report_v2,含分歧/缺失记录)
    → Judge(输出 judge_decision_v2: 最终类别 / 弃权 / 强制映射)
    → Gold Label 仅在预测完成后进入指标计算(严格防泄漏)
    → Optimizer(optimization_mode=apply 时在独立验证集上评测候选 Skill,
                通过才 promote 到 version store,失败自动 reject/回滚)

推理路径(run_test_evaluation):
  冻结测试域样本 + 当前 active Skill
  → 同样的 Coordinator/Specialist/Judge 链路
  → 全程不调用 Optimizer、不读任何训练标签之外的信号、不修改 Skill
```

Skill 版本化(审计 §6):`skills/` 目录中的每个 Skill 是一个**包**
(`SKILL.md` frontmatter + 正文 + references/assets/templates/scripts 等文本
资源),Optimizer **永不直接改写**基线目录;被接受的候选以**整包快照**
追加到 `skill_store/<name>/history.jsonl`,`active.json` 记录当前 active 版本,
`rollback(name, seq)` 可回到任意历史版本(0 = 基线)。非基线包会被物化到
`skill_store/<name>/active/`,保证 `scripts/` 能被真实加载为工具。

## 二、目录结构

```text
agents/               AutoGen Agent 包(固定 Skill 绑定,不再自由生成角色)
  coordinator.py      路由 Coordinator(routing_decision_v1)
  specialist.py       SpecialistAgent(固定 Skill + 样本任务)
  judge.py            Judge(消费 analysis_report_v2)
  optimizer.py        候选 Skill 生成器(永不直接改文件)
orchestration/
  prompt_builder.py   统一 Prompt 构建(基础提示词 + 角色 Skill + 契约)
  router.py           路由决策规范化 / 固定目录约束 / 兜底
  aggregator.py       多 Specialist 报告确定性聚合
  executor.py         InferenceRunner: 单样本完整前向推理
optimization/
  version_store.py    Skill 版本仓库(promote / rollback / 只读候选覆盖)
  validator.py        候选验证门控(对比 current active vs candidate)
evaluation/metrics.py Accuracy / Macro-F1 / 逐类 PRF / 混淆 / 弃权统计
datasets/dataset.py   Weibo21 / AMTCele 加载 + 域内确定性切分
skills/               固定 Skill 内容(coordinator_routing, claim_decomposition,
                      evidence_assessment, temporal_reasoning, judge_decision)
config.py             实验配置(train/test 域隔离、seed、label_schema)
trainner.py           训练/验证闭环 + 冻结测试评估
output_parsing.py     模型输出 JSON 稳健解析
```

## 三、快速开始

```bash
# 依赖(建议 conda 环境)
pip install -r requirements.txt

# 密钥只从环境变量读取(审计 §3.7):不要在代码里写 API Key
export DEEPSEEK_API_KEY=sk-...          # Windows: set DEEPSEEK_API_KEY=sk-...
export DEEPSEEK_MODEL=deepseek-v4-flash # 可选,默认 deepseek-v4-flash

# 基线训练(不调用 Optimizer,最省成本;先验证链路与标签映射)
python main.py --dataset weibo21 --mode train --epochs 1 --batch-size 8

# Optimizer 只出诊断与候选建议(不改任何文件)
python main.py --dataset weibo21 --mode train --epochs 1 --optimization suggest

# 完整优化闭环:候选过验证门控才晋升,失败自动回滚
python main.py --dataset weibo21 --mode train --epochs 3 --optimization apply

# 冻结测试域最终评估
python main.py --dataset weibo21 --mode test

# 运行测试(需要 autoagent 等装有 autogen 0.7.x 的环境)
python -m pytest -q
```

输出默认写入 `outputs/<dataset>-<run_id>/`(`predictions.jsonl` 逐样本完整
trace,`batches.jsonl` 批次日志与指标,`test/summary.json` 测试汇总)。

## 四、实验隔离与标签语义(务必阅读)

- `train_domains` 与 `test_domains` 默认**互不重叠**(跨域实验),Config 构造时
  会强制校验;验证集从训练域内按 `seed` 确定性切 `validation_ratio`。
- 标签语义必须显式:Weibo21 默认 `mapping = {REAL: 0, FAKE: 1}`(MDFEND/MFND
  惯例,已抽样核对;正式实验前请对照数据来源确认);AMTCele 默认
  `{REAL: "legit", FAKE: "fake"}`。若你的数据语义不同,请在 `Config` 传入
  `label_schema` 覆盖,不要改提示词。
- 一次实验请在固定 `seed` 下多跑几遍,报告均值 ± 标准差;Bootstrap 置信区间
  与显著性检验属于论文阶段扩展(见审计 §8 阶段 4)。
- 诚实记录**输出解析失败率**(`predictions.jsonl` 的 `errors` 字段):模型输出
  不合法时记 `parse_error`,不悄悄丢弃样本。

## 五、已知边界(有意为之,便于论文交代)

- 当前 Specialist 动作空间 = 3 个固定 Skill(claim_decomposition /
  evidence_assessment / temporal_reasoning);扩展只需:新增 `skills/<名>/SKILL.md`
  → 加入 `orchestration/prompt_builder.py` 的 `SPECIALIST_SKILL_NAMES`。
- LiveFact / AdvFake 的加载器尚未接入(`prompts.py` 已含对应数据集对齐规则),
  Weibo21 与 AMTCele 是当前默认实验对象。
- 校准(ECE/Brier)、显著性检验、消融脚本属阶段 4,未内置。
- 旧动态 spawn 架构(`agents.py / dynamic_workbench.py / tools.py /
  registry.py`)与 `skill_registry.py` 已删除,内容保留在 git 历史中可按需恢复。

## 六、Skill 包格式与 scripts→工具约定

每个 Skill 是一个目录,`SKILL.md` 用 YAML frontmatter 声明
`name / description / version / allowed_tools`;其余文件按约定分目录:

```text
skills/<name>/
├─ SKILL.md              # frontmatter + 正文指令(正文是优化器可整体替换的“规则”)
├─ references/…          # 只读参考规则 —— 自动注入系统提示词(<skill_resources>)
├─ templates/…           # 输出模板/示例 —— 自动注入系统提示词
├─ assets/…              # 数据类资源 —— 默认不注入,可被 scripts 读取
└─ scripts/…             # Python 脚本 —— 自动注册为该 Skill 绑定 Agent 的工具
```

### scripts → 工具注册规则
- `scripts/` 下每个 `.py` 由 `orchestration/skill_tools.py` 按路径 import;
- 注册对象 = 模块内**公开、可调用、带 docstring** 的函数(推荐用 `__all__`
  显式列出);下划线开头与从别处 import 进来的函数跳过;
- 参数请写类型注解或 `Annotated`(工具 Schema 需要);函数应纯函数化、
  输入非法时干净报错,禁止内嵌数据集答案/标签;
- 不同脚本导出同名函数会显式报错;Skill 晋升后 Agent `rebuild()` 自动
  重新加载新脚本,新工具立即生效。

### 优化器补丁协议(`proposed_change.patch`,字符串)
- **纯正文**:整段视为对 `instructions` 的完整替换(旧协议,仍然有效);
- **JSON 对象字符串**:
  ```json
  {"instructions": "新的正文(可选)",
   "resources": {"references/rules.md": "新文本",      // 更新/新建文本文件
                 "scripts/helper.py": "新代码",          // 新增/更新脚本(自动成工具)
                 "assets/old.txt": null}}                // null = 删除该文件
  ```
- 代码侧强约束(`apply_package_patch`):路径必须包内相对、禁止 `SKILL.md`、
  禁止越界/二进制、`scripts/` 只收 `.py`、单文件 ≤ 200 KB;
  frontmatter 元数据(含 `version`)永远由运行时维护,Optimizer 不可写;
- 候选仍须在独立验证集上**严格更优**才会被 promote;被 `scripts/` 引入的
  代码与系统进程同权限运行,投稿前请人工 review 每次晋升的脚本 diff
  (`skill_store/<name>/history.jsonl`)。
- **脚本自检前置门**:候选进入端到端评测前,`scripts/` 的每个 `.py` 会先做
  语法编译 + import/工具注册冒烟,失败直接拒收(不浪费 API、不带病评测);
  候选评测/晋升任一步异常都会**优雅降级为拒收并记录**,不会中断训练;
  Agent 重建采用“先建新、成功后再换掉旧”的策略,失败时旧 Skill 链路仍可用。
