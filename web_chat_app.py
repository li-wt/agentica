#!/usr/bin/env python3
"""
Agentica 塑化行业智能助手 - 简化版
专业服务于高分子材料和塑料加工领域
"""
import os
import streamlit as st
import time
import json
import re
from textwrap import dedent
from typing import Iterator, Optional, Dict, List
from datetime import datetime

from pydantic import BaseModel, Field

from agentica import Agent, logger
from agentica.workflow import Workflow
from agentica import RunResponse, RunEvent, SqlWorkflowStorage
from agentica.tools.baidu_search_tool import BaiduSearchTool
from agentica.tools.wikipedia_tool import WikipediaTool
from agentica.tools.calculator_tool import CalculatorTool
from agentica.tools.run_python_code_tool import RunPythonCodeTool
from agentica.tools.file_tool import FileTool
from agentica.tools.random_tool import RandomTool
from agentica.tools.text_processor_tool import TextProcessorTool
from agentica.tools.weather_tool import WeatherTool
from agentica.tools.datetime_tool import DateTimeTool

# 获取模型实例
from agentica import LocalChat
model_name = LocalChat

def parse_json_response(content: str) -> dict:
    """
    解析可能包含markdown代码块的JSON响应

    Args:
        content: 原始响应内容

    Returns:
        解析后的字典

    Raises:
        json.JSONDecodeError: JSON解析失败
    """
    if not content:
        raise json.JSONDecodeError("Empty content", "", 0)

    # 使用正则表达式提取JSON内容
    # 匹配 ```json...``` 或 ```...``` 代码块，或者直接的JSON
    patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",  # ```json ... ```
        r"```\s*\n?(.*?)\n?\s*```",  # ``` ... ```
        r"^(.*)$",  # 直接的JSON内容
    ]

    cleaned_content = content.strip()
    original_content = cleaned_content

    for pattern in patterns:
        match = re.search(pattern, cleaned_content, re.DOTALL | re.MULTILINE)
        if match:
            extracted = match.group(1).strip()
            # 验证提取的内容是否看起来像JSON
            if extracted.startswith(("{", "[")):
                cleaned_content = extracted
                break

    # 记录清理过程
    if cleaned_content != original_content:
        logger.info(f"[JSON_PARSE] 清理前: {original_content[:100]}...")
        logger.info(f"[JSON_PARSE] 清理后: {cleaned_content[:100]}...")

    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        logger.error(f"[JSON_PARSE] JSON解析失败: {e}")
        logger.error(f"[JSON_PARSE] 尝试解析的内容: {cleaned_content}")
        raise


class SmartTaskDecomposer(Workflow):
    """塑化行业智能助手 - 简化版"""

    description: str = (
        "专业的塑化行业智能助手，能够识别用户意图并提供专业解答或搜索最新信息"
    )

    # 只保留两个核心Agent
    intent_analyzer: Optional[Agent] = None
    expert_agent: Optional[Agent] = None  # 塑化行业专家

    def __init__(self, **data):
        super().__init__(**data)

        # 意图识别器 - 专门识别塑化行业相关意图
        self.intent_analyzer = Agent(
            model=model_name(),
            session_id=self.session_id,
            instructions=[
                dedent(
                    """
                你是一个专业的塑化行业意图识别专家，深度了解高分子材料、塑料加工、化工材料等领域。
                
                🎯 **核心职责**：识别用户的真实意图，判断是否需要搜索最新信息。
                
                📋 **意图判断原则**：
                
                **🔍 需要搜索的情况**：
                1. **最新市场信息** - 价格行情、市场动态、新产品信息等
                2. **实时数据** - 当前价格、最新标准、新技术发展等  
                3. **实时新闻和资讯** - 最新新闻、当前热点、股价汇率等
                4. **用户明确要求搜索** - 包含"搜索"、"查询"、"最新"等关键词
                
                **💡 无需搜索的情况**：
                1. **塑化行业基础知识** - 材料特性、工艺原理、基础概念等
                2. **历史知识** - 成熟的工艺技术、材料基础知识等
                3. **计算类问题** - 密度计算、体积换算等（可用计算工具）
                4. **概念解释** - 专业术语定义、原理说明等
                5. **日期时间信息** - 今天几号、现在几点、当前时间等（有专用工具）
                6. **天气信息** - 天气查询（有专用工具）
                
                **⚠️ 特别注意**：
                - 日期、时间查询直接使用DateTimeTool，无需搜索
                - 天气查询直接使用WeatherTool，无需搜索
                - 只有确实需要最新网络信息的问题才使用搜索
                - 能用专用工具解决的问题优先使用工具
                
                输出格式必须是JSON：
                {
                    "user_intent": "用户的真实意图描述",
                    "intent_category": "塑化专业知识/最新信息查询/常识性实时信息/计算分析",
                    "need_search": true/false,
                    "search_keywords": "如果需要搜索，提供关键词",
                    "reasoning": "判断理由（为什么需要或不需要搜索）"
                }
                """
                ),
            ],
        )

        # 塑化行业专家 - 集成搜索工具的全能专家
        self.expert_agent = Agent(
            model=model_name(),
            session_id=self.session_id,
            tools=[
                BaiduSearchTool(),
                WikipediaTool(), 
                CalculatorTool(),
                RunPythonCodeTool(),
                FileTool(),
                RandomTool(),
                TextProcessorTool(),
                WeatherTool(),
                DateTimeTool(),
            ],
            instructions=[
                dedent(
                    """
                你是塑化行业的资深专家，拥有20年高分子材料、塑料加工、化工技术经验。
                你深度了解塑化行业的各个方面：
                
                📋 **专业领域**：
                - 高分子材料特性与应用（PP、PE、PVC、PS、ABS等）
                - 塑料加工工艺与设备（注塑、挤出、吹塑、热成型等）
                - 化工原料与配方设计
                - 质量控制与检测标准
                - 成本分析与市场预测
                - 环保法规与可持续发展
                
                🔧 **可用工具**：
                - BaiduSearchTool: 搜索最新行业信息
                - WikipediaTool: 查询权威技术资料
                - CalculatorTool: 进行数值计算
                - RunPythonCodeTool: 执行复杂计算和数据分析
                - FileTool: 文件读写操作
                - RandomTool: 生成测试数据
                - TextProcessorTool: 文本处理和格式化
                - WeatherTool: 查询天气信息
                - DateTimeTool: 获取准确的当前日期和时间
                
                🎯 **工作原则**：
                1. **日期时间查询** - 必须优先使用DateTimeTool获取准确信息
                2. **天气查询** - 使用WeatherTool获取天气信息
                3. **需要最新网络信息时** - 使用搜索工具获取实时数据
                4. **专业知识问答** - 基于深厚的行业经验直接回答
                5. **数据计算** - 使用计算工具进行精确计算
                6. **文件操作** - 根据需要保存或处理文件
                
                ⚠️ **强制要求**：
                - 涉及日期、时间的任何问题，必须先调用DateTimeTool获取当前准确时间
                - 回答中必须包含通过工具获取的实时日期时间信息
                - 不能基于训练数据推测日期时间，必须使用工具获取
                - 即使是简单的日期时间问题，也要调用对应的工具函数
                
                💡 **回答特点**：
                - 专业、准确、实用
                - 结合理论与实践
                - 关注行业最新发展
                - 提供可操作的建议
                """
                ),
            ],
        )

    def run(self, user_request: str) -> Iterator[RunResponse]:
        """简化的智能处理流程"""
        
        logger.info(f"[TASK] 开始处理用户请求: {user_request}")
        self.run_id = f"run_{int(time.time())}"

        # 第一步：意图识别
        yield RunResponse(run_id=self.run_id, content="🎯 正在分析您的需求...\n")
        
        intent_data = None
        try:
            intent_response = self.intent_analyzer.run(user_request)
            if intent_response and intent_response.content:
                intent_data = parse_json_response(intent_response.content)
                logger.info(f"[INTENT] 意图识别成功: {intent_data.get('user_intent', '未知')}")
                
                # 显示意图识别结果
                yield RunResponse(
                    run_id=self.run_id, 
                    content=f"✅ 识别意图: {intent_data.get('user_intent', '未知')}\n"
                )
                
                if intent_data.get('need_search', False):
                    yield RunResponse(
                        run_id=self.run_id, 
                        content=f"🔍 需要搜索最新信息: {intent_data.get('search_keywords', '')}\n"
                    )
                else:
                    yield RunResponse(
                        run_id=self.run_id, 
                        content="💡 基于专业知识直接回答\n"
                    )
                    
        except Exception as e:
            logger.error(f"[INTENT] 意图识别失败: {e}")
            intent_data = {"user_intent": "通用问答", "need_search": False}

        # 第二步：执行回答
        yield RunResponse(run_id=self.run_id, content="⚙️ 正在处理您的问题...\n")
        
        try:
            # 构建专家输入
            if intent_data and intent_data.get('need_search', False):
                expert_input = f"""
                用户问题: {user_request}
                意图分析: {intent_data.get('user_intent', '')}
                搜索建议: {intent_data.get('search_keywords', '')}
                
                请先使用搜索工具获取最新信息，然后结合您的专业知识给出全面的回答。
                """
            else:
                expert_input = f"""
                用户问题: {user_request}
                意图分析: {intent_data.get('user_intent', '') if intent_data else '通用问答'}
                
                请基于您的塑化行业专业知识直接回答。
                """
            
            # 专家回答
            expert_response = self.expert_agent.run(expert_input)
            
            if expert_response and expert_response.content:
                yield RunResponse(run_id=self.run_id, content="✅ 处理完成！\n")
                yield RunResponse(run_id=self.run_id, content=f"## 🎯 **专家解答**\n\n{expert_response.content}")
            else:
                yield RunResponse(run_id=self.run_id, content="❌ 处理失败，请重试")
                
        except Exception as e:
            logger.error(f"[EXECUTE] 执行失败: {e}")
            yield RunResponse(run_id=self.run_id, content="❌ 处理过程中出现错误，请重试")
        
        logger.info(f"[TASK] 任务处理流程完成")


class SpecialCommandHandler:
    """特殊指令处理器"""

    @staticmethod
    def handle_recent_changes(agent) -> str:
        """处理 @Recent Changes 指令"""
        try:
            if hasattr(agent, "storage") and agent.storage:
                recent_sessions = agent.storage.get_all_session_ids()[-5:]
                changes_info = []

                for session_id in recent_sessions:
                    changes_info.append(
                        f"会话 {session_id} 在 {datetime.now().strftime('%Y-%m-%d %H:%M')} 有活动"
                    )

                if changes_info:
                    return "📋 **最近变更**:\n\n" + "\n".join(
                        [f"• {info}" for info in changes_info]
                    )
                else:
                    return "🔍 暂时没有发现最近的变更记录。"
            else:
                return "💡 当前会话没有启用存储功能，无法查看历史变更。"
        except Exception as e:
            return f"❌ 处理最近变更时出现错误: {str(e)}"

    @staticmethod
    def handle_help() -> str:
        """处理帮助指令"""
        return """
🤖 **Agentica 智能任务拆分助手**

### 🧠 **核心能力**
- **智能意图识别**: 自动理解您的需求复杂度
- **动态任务拆分**: 将复杂任务分解为可执行步骤
- **多Agent协同**: 调用专业Agent完成不同类型任务
- **智能结果整合**: 统一输出完整、结构化的答案

### 🛠️ **支持的任务类型**
- 📊 **研究分析**: 市场调研、技术分析、趋势预测
- 📋 **计划制定**: 学习计划、健身计划、项目规划
- 🧮 **数据计算**: 复杂计算、数据分析、可视化
- 📝 **内容创作**: 报告撰写、文档整理、内容优化
- 🔍 **信息整合**: 多源信息收集、对比分析

### ⚡ **使用示例**
- "分析人工智能在医疗领域的应用现状和发展趋势"
- "制定一个月的减肥健身计划，包括饮食和运动安排"
- "研究比特币价格影响因素并预测未来走势"
- "比较不同编程语言的优缺点并推荐学习路径"

### 🎯 **特殊功能**
- `@Recent Changes` - 查看最近变更
- `@Help` - 显示帮助信息
- 支持简单任务直接处理
- 支持复杂任务智能拆分

开始您的智能任务吧！
        """


def main():
    # 页面配置
    st.set_page_config(
        page_title="塑化行业智能助手 - Agentica",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 主标题
    st.title("🧪 塑化行业智能助手")
    st.markdown(
        "##### 🧬 专业服务高分子材料 | 🏭 塑料加工工艺 | 🤖 多Agent协同 | 📊 行业智能分析"
    )
    st.markdown(
        "*专注塑化行业的AI智能助手，为您提供材料技术、工艺优化、市场分析等专业服务*"
    )

    # 侧边栏 - 简化设置
    with st.sidebar:
        st.header("⚙️ 设置")

        # 清空对话按钮
        if st.button("🗑️ 清空对话", type="primary", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": get_welcome_message()}
            ]
            st.rerun()

        st.divider()

        # 系统状态
        st.subheader("🔌 系统状态")

        # 检查 model_name 可用性
        try:
            test_model = model_name()
            st.success("✅ model_name 模型可用")
        except Exception as e:
            st.error(f"❌ model_name 模型不可用: {str(e)}")
            st.warning("请检查本地模型服务是否正常运行")

        # 工具提示
        with st.expander("🛠️ 可用工具"):
            st.markdown(
                """
            **内置工具集**:
            - 🔍 **搜索工具**: 网络搜索、维基百科
            - 🧮 **计算工具**: 数学计算、Python代码
            - 🎲 **随机工具**: 随机数生成、随机选择
            - 📝 **文本工具**: 文本分析、信息提取
            - 📁 **文件工具**: 文件读写、文档管理
            - 🌤️ **天气工具**: 实时天气查询
            
            所有工具会根据任务需求自动调用，无需手动选择。
            """
            )

        # 高级设置
        with st.expander("⚙️ 高级设置"):
            debug_mode = st.checkbox("🐛 调试模式", value=False)
            show_thinking = st.checkbox("💭 显示思维过程", value=True)

        # 示例任务
        with st.expander("📋 塑化行业示例任务"):
            example_tasks = [
                "PE材料的密度和熔融指数对注塑工艺的影响",
                "PP改性材料的配方设计要点",
                "ABS塑料的耐热性能如何提升？",
                "分析当前塑料原料市场价格趋势",
                "制定注塑机参数优化方案",
                "比较PVC和PE在管材应用中的优缺点",
            ]

            for task in example_tasks:
                if st.button(
                    f"💡 {task}", key=f"example_{hash(task)}", use_container_width=True
                ):
                    st.session_state.messages.append({"role": "user", "content": task})
                    st.rerun()

    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)

    # 初始化消息历史
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": get_welcome_message()}
        ]

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 用户输入处理
    if prompt := st.chat_input("💬 请描述您的需求..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 显示用户消息
        with st.chat_message("user"):
            st.write(prompt)

        # 处理用户请求
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # 检查特殊指令
            if re.search(r"@recent.*changes", prompt, re.IGNORECASE):
                full_response = SpecialCommandHandler.handle_recent_changes(None)
                response_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            elif re.search(r"@help|帮助", prompt, re.IGNORECASE):
                full_response = SpecialCommandHandler.handle_help()
                response_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            else:
                # 使用智能任务拆分处理所有请求
                try:
                    if show_thinking:
                        st.info("🧠 启用智能处理模式，正在分析任务...")

                    decomposer = SmartTaskDecomposer(
                        session_id=f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        storage=SqlWorkflowStorage(
                            table_name="smart_task_workflows",
                            db_file="outputs/smart_task_workflows.db",
                        ),
                    )

                    # 真正的流式处理任务
                    import time

                    for response in decomposer.run(prompt):
                        if response.content:
                            full_response += response.content
                            # 实时更新显示，添加打字机效果
                            response_placeholder.markdown(full_response + "▌")
                            # 短暂延迟以产生流式效果
                            time.sleep(0.01)

                    # 移除光标
                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )

                except Exception as e:
                    error_msg = f"❌ 处理请求时出现错误: {str(e)}"
                    if debug_mode:
                        error_msg += f"\n\n**调试信息**: {repr(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # 底部功能按钮
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 重新开始", type="secondary", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": get_welcome_message()}
            ]
            st.rerun()

    with col2:
        if st.button("📊 查看统计", type="secondary", use_container_width=True):
            st.info(
                f"""
**会话统计**:
- 消息数量: {len(st.session_state.messages)}
- 用户消息: {len([m for m in st.session_state.messages if m['role'] == 'user'])}
- 助手回复: {len([m for m in st.session_state.messages if m['role'] == 'assistant'])}
- 调试模式: {'开启' if debug_mode else '关闭'}
            """
            )

    with col3:
        if st.button("❓ 帮助", type="secondary", use_container_width=True):
            help_msg = SpecialCommandHandler.handle_help()
            st.info(help_msg)


def get_welcome_message() -> str:
    """获取欢迎消息"""
    return """
🎉 **欢迎使用塑化行业智能助手！**

### 🧪 **塑化行业专业服务**
我是专门为高分子材料和塑料加工领域设计的AI智能助手，深度了解：
- **材料特性**: 密度、熔融指数、强度、耐热性等关键性能参数
- **加工工艺**: 注塑、挤出、吹塑、混炼等工艺技术
- **质量控制**: 检测标准、质量管理、缺陷分析
- **市场动态**: 原料价格、供需情况、行业趋势

### 🧠 **智能处理能力**
我会自动分析您的塑化行业需求，智能选择最佳处理方式：
- **简单咨询**: 直接提供专业解答（材料参数查询、工艺问题等）
- **复杂分析**: 自动拆分任务，深度分析（配方设计、工艺优化、市场研究等）

### 🛠️ **专业工具集**
- 🔍 **技术搜索** - 专业文献、行业标准、技术资料查询
- 🧮 **工程计算** - 材料性能计算、工艺参数优化
- 📝 **专业写作** - 技术文档、工艺说明、产品介绍
- 📊 **数据分析** - 性能对比、成本分析、市场调研
- 📁 **资料管理** - 技术文档整理、数据统计
- 🌡️ **环境监控** - 生产环境条件查询

### 💼 **适用人群**
- **工程师**: 材料选择、工艺设计、技术问题解决
- **研发人员**: 配方开发、性能优化、新材料研究
- **采购人员**: 原料行情、供应商评估、成本分析
- **销售人员**: 产品介绍、技术支持、市场分析
- **管理人员**: 决策支持、行业趋势、竞争分析

### 💡 **使用建议**
直接描述您的塑化行业需求即可，例如：
- "PE材料的密度和注塑工艺的关系"
- "如何提高PP材料的耐热性能？"
- "当前ABS原料市场价格趋势分析"
- "制定一套注塑参数优化方案"

开始您的塑化行业智能咨询之旅吧！
    """


if __name__ == "__main__":
    main()
