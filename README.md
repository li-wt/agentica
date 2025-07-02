# Agentica 🤖

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/logo.png" height="150" alt="Agentica Logo">
  
  [![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
  [![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
  [![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
  [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
  [![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
  
  **一个轻量级、模块化的 AI Agent 框架**
  
  [🇨🇳中文文档](README.md) | [🌐English](README_EN.md) | [🇯🇵日本語](README_JP.md)
</div>

---

## 📖 简介

**Agentica** 是一个专为中文用户打造的轻量级、模块化 AI Agent 框架。它专注于构建智能、具备反思能力、可协作的多模态 AI Agent，提供丰富的工具集成、多模型支持和灵活的工作流编排能力。

### 🎯 核心优势

- **🚀 简单易用**：简洁直观的 API 设计，5 分钟快速上手
- **🔧 功能全面**：从单 Agent 到多 Agent 协作，从简单对话到复杂工作流
- **🇨🇳 中文优化**：为中文用户深度优化，支持国内主流大模型
- **🛠️ 开箱即用**：内置 40+ 工具和丰富示例，覆盖常见应用场景
- **🔄 自我进化**：支持反思和记忆能力，Agent 可以自我改进

## ✨ 主要特性

### 🤖 Agent 能力
- **智能编排**：支持 Reflection（反思）、Plan and Solve（计划执行）、RAG 等高级能力
- **记忆管理**：短期记忆 + 长期记忆，让 Agent 更智能
- **多模态支持**：文本、图片、音频、视频多模态输入处理
- **自我进化**：具备反思和增强记忆能力

### 🔌 模型集成
支持国内外主流大模型：
- **国外模型**：OpenAI、Azure、Anthropic、Together 等
- **国内模型**：DeepSeek、月之暗面、智谱AI、豆包、Yi 等
- **本地模型**：Ollama 等本地部署方案

### 🛠️ 工具生态
内置 40+ 实用工具：
- **搜索工具**：网页搜索、学术搜索、新闻搜索
- **文件工具**：PDF 解析、Excel 处理、文档转换
- **图像工具**：图像生成、OCR 识别、背景去除
- **实用工具**：天气查询、计算器、Shell 命令
- **自定义工具**：轻松扩展自己的工具

### 🏗️ 架构特性
- **Multi-Agent 协作**：支持多 Agent 团队协作和任务委托
- **Workflow 工作流**：复杂任务自动拆解和串行执行
- **MCP 协议**：支持 Model Context Protocol，标准化模型交互
- **存储系统**：SQL + 向量数据库双重存储方案

## 🚀 快速开始

### 安装

```bash
# 使用 pip 安装
pip install -U agentica

# 或从源码安装
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install .
```

### 环境配置

1. 设置 API Key（任选其一）：
```bash
export OPENAI_API_KEY="your_api_key"
export DEEPSEEK_API_KEY="your_api_key"
export MOONSHOT_API_KEY="your_api_key"
export ZHIPUAI_API_KEY="your_api_key"
```

2. 或创建 `~/.agentica/.env` 文件：
```env
MOONSHOT_API_KEY=your_api_key
SERPER_API_KEY=your_serper_api_key
```

### 基础示例

#### 1. 简单对话
```python
from agentica import Agent, Moonshot

# 创建一个基础 Agent
agent = Agent(model=Moonshot())
agent.print_response("你好，介绍一下你自己")
```

#### 2. 天气查询
```python
from agentica import Agent, Moonshot, WeatherTool

# 创建带工具的 Agent
agent = Agent(
    model=Moonshot(), 
    tools=[WeatherTool()], 
    add_datetime_to_instructions=True
)
agent.print_response("明天北京天气怎么样？")
```

#### 3. 自定义工具
```python
from agentica import Agent, OpenAIChat, Tool

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(name="calculator")
        self.register(self.calculate)
    
    def calculate(self, expression: str) -> str:
        """计算数学表达式"""
        try:
            result = eval(expression)
            return f"计算结果：{result}"
        except Exception as e:
            return f"计算错误：{str(e)}"

# 使用自定义工具
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[CalculatorTool()]
)
agent.print_response("帮我计算 (123 + 456) * 789")
```

## 📚 功能示例

### 🔍 RAG 知识问答
```python
from agentica import Agent, OpenAIChat, KnowledgeBase

# 创建知识库
kb = KnowledgeBase(
    name="my_docs",
    path="./documents"
)

# 创建 RAG Agent
agent = Agent(
    model=OpenAIChat(),
    knowledge_base=kb,
    use_tools=True
)

agent.print_response("根据我的文档，总结一下产品特性")
```

### 👥 多 Agent 协作
```python
from agentica import Agent, Team, OpenAIChat

# 创建专业化 Agent
researcher = Agent(
    name="研究员",
    role="负责收集和分析信息",
    model=OpenAIChat(),
    tools=[WebSearchTool()]
)

writer = Agent(
    name="作家",
    role="负责撰写文章",
    model=OpenAIChat()
)

# 创建团队
team = Team([researcher, writer])
team.print_response("写一篇关于 AI 发展趋势的文章")
```

### 🔄 工作流编排
```python
from agentica import Workflow, Agent, OpenAIChat

# 创建工作流
workflow = Workflow(
    name="文章生成流程",
    description="自动化文章生成工作流"
)

# 添加工作步骤
workflow.add_step("research", "收集相关资料")
workflow.add_step("outline", "制作文章大纲")
workflow.add_step("write", "撰写文章内容")
workflow.add_step("review", "审核和优化")

# 执行工作流
agent = Agent(model=OpenAIChat(), workflow=workflow)
agent.run("写一篇关于人工智能的技术文章")
```

## 🎨 应用场景

### 💼 商业应用
- **智能客服**：24/7 自动回复，支持多轮对话
- **内容创作**：自动生成文章、报告、营销文案
- **数据分析**：自动分析数据并生成洞察报告
- **知识管理**：企业知识库问答和检索

### 🎓 教育科研
- **学习助手**：个性化学习指导和答疑
- **论文助手**：文献检索、摘要生成、写作辅导
- **研究工具**：数据收集、分析和可视化

### 🏠 个人助手
- **日程管理**：智能安排和提醒
- **信息整理**：自动收集和分类信息
- **生活助手**：天气查询、路线规划等

## 🏗️ 系统架构

<div align="center">
    <img src="https://github.com/shibing624/agentica/blob/main/docs/agentica_architecture.png" alt="Agentica Architecture" width="800"/>
</div>

Agentica 采用模块化设计，主要包括以下核心组件：

### 核心组件
- **Agent Core**: 核心控制模块，负责 Agent 的创建和管理
- **Model Integration**: 模型接入层，支持多种 LLM 模型接口
- **Tools System**: 工具调用系统，提供丰富的工具调用能力
- **Memory Management**: 记忆管理，实现短期和长期记忆功能
- **Multi-Agent Collaboration**: 多 Agent 协作，实现团队协作和任务委托
- **Workflow Orchestration**: 工作流编排，支持复杂任务的拆解和执行

## 📋 版本更新

- **v1.0.10** (2025/06/19): 支持思考过程流式输出，适配所有推理模型
- **v1.0.6** (2025/05/19): 新增 MCP StreamableHttp 支持
- **v1.0.0** (2025/04/21): 支持 MCP 工具调用，兼容多种 MCP Server
- **v0.2.3** (2024/12/29): 支持智谱AI API，包括免费模型
- **v0.2.0** (2024/12/25): 支持多模态模型，升级 Workflow 功能
- **v0.1.0** (2024/07/02): 首个版本发布

[查看完整更新日志](https://github.com/shibing624/agentica/releases)

## 🛠️ 开发指南

### 项目结构
```
agentica/
├── agentica/           # 核心代码
│   ├── agent.py       # Agent 核心实现
│   ├── model/         # 模型集成
│   │   ├── openai/    # OpenAI 模型
│   │   ├── deepseek/  # DeepSeek 模型
│   │   ├── moonshot/  # Moonshot 模型
│   │   └── ...
│   ├── tools/         # 工具系统
│   ├── memory.py      # 记忆管理
│   ├── workflow.py    # 工作流
│   ├── storage/       # 存储系统
│   ├── vectordb/      # 向量数据库
│   └── utils/         # 工具函数
├── examples/          # 示例代码
│   ├── 01_llm_demo.py
│   ├── 14_custom_tool_demo.py
│   └── ...
├── tests/            # 测试代码
└── docs/             # 文档资源
```

### 自定义开发

#### 创建自定义工具
```python
from agentica import Tool

class MyTool(Tool):
    def __init__(self):
        super().__init__(name="my_tool")
        self.register(self.my_function)
    
    def my_function(self, param: str) -> str:
        """工具功能描述"""
        # 实现你的逻辑
        return "处理结果"
```

#### 扩展模型支持
```python
from agentica.model import LLM

class MyModel(LLM):
    def __init__(self):
        super().__init__()
    
    def response(self, messages):
        # 实现模型调用逻辑
        pass
```

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

### 开发环境设置
```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install -e .
pip install -r requirements-dev.txt
```

### 运行测试
```bash
python -m pytest tests/
```

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 🙋‍♂️ 常见问题

### Q: 如何选择合适的模型？
A: 
- **DeepSeek**: 性价比高，适合大多数场景
- **Moonshot**: 长文本处理能力强
- **OpenAI**: 综合能力最强，但成本较高
- **智谱AI**: 中文理解好，有免费额度

### Q: 工具调用失败怎么办？
A: 
1. 检查 API Key 是否正确设置
2. 确认网络连接正常
3. 查看工具文档和参数格式
4. 开启 debug 模式查看详细日志

### Q: 如何优化 Agent 性能？
A:
1. 合理设置 prompt 和 instructions
2. 选择合适的模型和参数
3. 使用记忆功能减少重复计算
4. 合理使用工具，避免过度调用

## 💬 社区与支持

- **GitHub Issues**: [提交问题](https://github.com/shibing624/agentica/issues)
- **讨论区**: [GitHub Discussions](https://github.com/shibing624/agentica/discussions)
- **邮箱**: xuming624@qq.com

## 🌟 致谢

感谢所有为 Agentica 做出贡献的开发者和用户！

如果这个项目对你有帮助，请给我们一个 ⭐️ Star！

---

<div align="center">
  Made with ❤️ by the Agentica Team
</div> 