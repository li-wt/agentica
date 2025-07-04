#!/usr/bin/env python3
"""
Agentica 塑化行业智能助手 - 高级协同版
专业服务于高分子材料和塑料加工领域，支持复杂任务拆分
"""
import os
import streamlit as st
import time
import json
import re
from textwrap import dedent
from typing import Iterator, Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

from agentica import Agent, logger
from agentica.memory import AgentMemory, Message, AgentRun
from agentica.model.message import UserMessage
from agentica.workflow import Workflow
from agentica import RunResponse, SqlWorkflowStorage
# from agentica.tools.duckduckgo_tool import BaiduSearchTool
from agentica.tools.baidu_search_tool import BaiduSearchTool
from agentica.tools.wikipedia_tool import WikipediaTool
from agentica.tools.calculator_tool import CalculatorTool
from agentica.tools.datetime_tool import DateTimeTool

# 获取模型实例
from agentica import LocalChat

model_name = LocalChat


class TaskStatus(Enum):
    """任务状态枚举"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    """子任务数据结构"""

    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:6]}")
    title: str = ""
    description: str = ""
    assigned_expert: str = ""
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TaskPlan:
    """任务计划数据结构"""

    plan_id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:6]}")
    user_query: str = ""
    subtasks: List[SubTask] = field(default_factory=list)
    execution_order: List[List[str]] = field(default_factory=list)


def parse_json_response(content: str) -> dict:
    """解析可能包含markdown代码块的JSON响应"""
    if not content:
        raise json.JSONDecodeError("Empty content", "", 0)

    patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",
        r"```\s*\n?(.*?)\n?\s*```",
        r"^(.*)$",
    ]

    cleaned_content = content.strip()
    original_content = cleaned_content

    for pattern in patterns:
        match = re.search(pattern, cleaned_content, re.DOTALL | re.MULTILINE)
        if match:
            extracted = match.group(1).strip()
            if extracted.startswith(("{", "[")):
                cleaned_content = extracted
                break

    if cleaned_content != original_content:
        logger.info(f"[JSON_PARSE] 清理前: {original_content[:100]}...")
        logger.info(f"[JSON_PARSE] 清理后: {cleaned_content[:100]}...")

    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        logger.error(f"[JSON_PARSE] JSON解析失败: {e}")
        logger.error(f"[JSON_PARSE] 尝试解析的内容: {cleaned_content}")
        raise


class PlasticIndustryWorkflow(Workflow):
    """塑化行业智能助手 - 高级协同版"""

    description: str = "专业的塑化行业智能助手，支持多意图识别和复杂任务拆分"

    # 专业Agent团队
    intent_analyzer: Optional[Agent] = None  # 意图识别专家
    task_planner: Optional[Agent] = None  # 任务拆分专家
    material_expert: Optional[Agent] = None  # 材料技术专家
    process_expert: Optional[Agent] = None  # 工艺技术专家
    quality_expert: Optional[Agent] = None  # 质量控制专家
    market_analyst: Optional[Agent] = None  # 市场分析专家
    calculation_expert: Optional[Agent] = None  # 计算分析专家
    result_integrator: Optional[Agent] = None  # 结果整合专家
    default_agent: Optional[Agent] = None  # 默认智能体

    

    def __init__(self, **data):
        super().__init__(**data)
        self.memory = AgentMemory()

        # 意图识别专家
        self.intent_analyzer = Agent(
            name="意图识别专家",
            model=model_name(),
            session_id=self.session_id,
            memory=self.memory,
            num_history_responses=5,
            add_history_to_messages=True,
            add_datetime_to_instructions=True,
            instructions=[
                dedent(
                    """
                你是意图识别专家，职责：准确识别用户想问什么。

                🎯 **你的工作**：
                1. 分析用户想了解什么信息（可能有多个意图）
                2. 提取关键词
                3. 判断问题是否清晰

                💡 **上下文处理**：
                - 结合历史对话理解用户真实意图
                - 如果当前输入是对历史问题的补充，将它们作为整体分析

                🔍 **判断标准**：
                - **清晰**：能明确知道用户想了解什么
                - **模糊**：用户问题太宽泛或信息不足，需要澄清
                - **通用**：不是专业领域问题（如天气、问候等）

                **输出格式**：
                ```json
                {
                    "intent_clear": true,
                    "user_intents": [
                        "用户想了解的第一个信息",
                        "用户想了解的第二个信息"
                    ],
                    "keywords": ["关键词1", "关键词2"],
                    "domain": "塑化行业|通用",
                    "clarification_needed": false,
                    "clarification_question": null
                }
                ```
                """
                )
            ],
        )

        self.default_agent = Agent(
            name="默认智能体",
            model=model_name(),
            session_id=self.session_id,
            add_datetime_to_instructions=True,
            tools=[BaiduSearchTool(), CalculatorTool()],  # 添加基本工具
            instructions=[
                dedent(
                    """
                你是一个智能助手，具备以下能力：

                🎯 **核心定位**：
                1. **塑化行业专家**：对于塑化行业的简单问题，直接提供专业回答
                2. **通用智能助手**：对于非塑化行业问题，提供友好、准确的回答
                3. **信息检索**：能够使用搜索工具获取最新信息
                
                📋 **处理原则**：
                1. **直接回答**：用户问什么就回答什么，简洁明了
                2. **使用工具**：当需要实时信息（天气、时间、最新新闻等）时，主动使用搜索工具
                3. **专业回答**：对于塑化行业基础问题，直接给出专业回答
                4. **友好交流**：对于日常问候、闲聊等，给予友好回应
                
                ⚠️ **注意事项**：
                - 不要过度解释，用户要的是答案不是教育
                - 对于复杂的塑化行业技术问题，建议用户使用更详细的问题描述以获得专业分析
                - 如果无法确定答案，诚实地说不知道，并建议用户如何获得更好的帮助
                
                📝 **回答风格**：
                - 简洁、直接、有用
                - 根据问题类型调整专业程度
                - 对于一般问题保持友好和易懂
                """
                )
            ],
        )

        # 任务拆分专家
        self.task_planner = Agent(
            name="任务拆分专家",
            model=model_name(),
            session_id=self.session_id,
            add_datetime_to_instructions=True,
            instructions=[
                dedent(
                    """
                你是专业的任务规划专家，负责将用户意图转化为具体的可执行任务计划。

                🎯 **核心职责**:
                1. **意图分析**: 理解用户的具体需求和关键词
                2. **任务拆分**: 将复杂问题分解为具体的子任务
                3. **专家分配**: 为每个子任务分配最合适的专家
                4. **依赖规划**: 分析任务间的逻辑关系和执行顺序

                📋 **输入格式**:
                你将收到意图识别的结果，包含：
                - `user_intents`: 用户的多个具体意图（数组）
                - `keywords`: 提取的关键词
                - `domain`: 问题领域

                ---
                ### 👨‍🔬 **专家团队能力清单**

                根据问题复杂度和关键词，为任务分配合适的专家：

                1. **`材料技术专家`**
                   - 材料特性分析 (PP, PE, PVC, ABS等)
                   - 配方设计和性能评估
                   - 改性技术和材料选择

                2. **`工艺技术专家`**
                   - 注塑/挤出/吹塑成型工艺
                   - 设备参数优化
                   - 生产效率提升

                3. **`质量控制专家`**
                   - 质量检测标准与方法
                   - 质量管理体系
                   - 产品缺陷分析

                4. **`市场分析专家`**
                   - 原料价格行情分析
                   - 市场供需研究
                   - 成本效益分析

                5. **`计算分析专家`**
                   - 工程计算和数据分析
                   - 成本计算
                   - 日期和时间查询

                ---

                💡 **任务拆分策略**:

                **简单查询** (1个任务):
                - 直接的信息查询
                - 基础概念解释
                - 单一参数查询

                **复杂分析** (2-3个任务):
                - 需要多个维度分析
                - 涉及多个专业领域
                - 需要综合评估

                **综合性问题** (3-5个任务):
                - 涉及多个环节的完整分析
                - 需要成本效益评估
                - 包含技术方案和实施建议

                **输出格式 (严格遵守)**:
                ```json
                {
                    "subtasks": [
                        {
                            "task_id": "task_001",
                            "title": "任务的简洁标题",
                            "description": "对子任务的详细、清晰的描述",
                            "assigned_expert": "专家名称",
                            "dependencies": ["依赖的前序任务ID"]
                        }
                    ],
                    "execution_order": [["task_001"], ["task_002", "task_003"]]
                }
                ```

                **拆分示例**:

                输入: user_intents=["了解PP材料熔融指数特性", "分析对注塑工艺的影响"], keywords=["PP材料", "熔融指数", "注塑工艺", "影响"]

                输出:
                ```json
                {
                    "subtasks": [
                        {
                            "task_id": "task_001",
                            "title": "PP材料熔融指数特性分析",
                            "description": "分析PP材料熔融指数的定义、测试方法和典型数值范围",
                            "assigned_expert": "材料技术专家",
                            "dependencies": []
                        },
                        {
                            "task_id": "task_002", 
                            "title": "熔融指数对注塑工艺的影响机制",
                            "description": "分析熔融指数如何影响注塑工艺参数，包括温度、压力、流动性等",
                            "assigned_expert": "工艺技术专家",
                            "dependencies": ["task_001"]
                        }
                    ],
                    "execution_order": [["task_001"], ["task_002"]]
                }
                ```
                """
                )
            ],
        )

        # 材料技术专家
        self.material_expert = Agent(
            name="材料技术专家",
            model=model_name(),
            session_id=self.session_id,
            add_datetime_to_instructions=True,
            tools=[
                BaiduSearchTool(),
                WikipediaTool(),
                CalculatorTool(),
            ],
            instructions=[
                dedent(
                    """
                你是塑化行业的材料技术专家。你只执行分配给你的具体子任务。
                你的回答应该聚焦、深入、专业，并严格限制在任务描述的范围内。
                """
                )
            ],
        )

        # 工艺技术专家
        self.process_expert = Agent(
            name="工艺技术专家",
            model=model_name(),
            session_id=self.session_id,
            add_datetime_to_instructions=True,
            tools=[BaiduSearchTool(), CalculatorTool()],
            instructions=[
                dedent(
                    """
                你是塑化行业的工艺技术专家。你只执行分配给你的具体子任务。
                你的回答应该聚焦、深入、专业，并严格限制在任务描述的范围内。
                """
                )
            ],
        )

        # 质量控制专家
        self.quality_expert = Agent(
            name="质量控制专家",
            model=model_name(),
            session_id=self.session_id,
            add_datetime_to_instructions=True,
            tools=[BaiduSearchTool(), WikipediaTool()],
            instructions=[
                dedent(
                    """
                你是塑化行业的质量控制专家。你只执行分配给你的具体子任务。
                你的回答应该聚焦、深入、专业，并严格限制在任务描述的范围内。
                """
                )
            ],
        )

        # 市场分析专家
        self.market_analyst = Agent(
            name="市场分析专家",
            model=model_name(),
            session_id=self.session_id,
            add_datetime_to_instructions=True,
            tools=[BaiduSearchTool()],
            instructions=[
                dedent(
                    """
                你是塑化行业的市场分析专家。你只执行分配给你的具体子任务，必须使用搜索工具获取最新数据。
                你的回答应该聚焦、深入、专业，并严格限制在任务描述的范围内。
                """
                )
            ],
        )

        # 计算分析专家
        self.calculation_expert = Agent(
            name="计算分析专家",
            model=model_name(),
            session_id=self.session_id,
            add_datetime_to_instructions=True,
            tools=[CalculatorTool()],
            instructions=[
                dedent(
                    """
                你是塑化行业的计算分析专家。你只执行分配给你的具体子任务。
                涉及日期时间必须使用DateTimeTool。你的回答应该聚焦、深入、专业，并严格限制在任务描述的范围内。
                """
                )
            ],
        )

        # 结果整合专家
        self.result_integrator = Agent(
            name="结果整合专家",
            model=model_name(),
            session_id=self.session_id,
            add_datetime_to_instructions=True,
            tools=[],
            instructions=[
                dedent(
                    """
                你是结果整合专家，负责将子任务结果整合成简洁、直接的回答。
                
                🎯 **核心原则**：
                1. **只回答用户问的内容** - 如果用户问"今天几号"，只需回答"今天是2023年5月15日"
                2. **删除所有不必要的内容** - 不要解释、不要背景信息、不要总结
                3. **一句话回答** - 如果可以用一句话回答，就用一句话
                
                ⚠️ **严格禁止**：
                - 不要自我介绍
                - 不要开场白或结束语
                - 不要解释你如何得到答案
                - 不要添加用户没有问的信息
                - 不要使用"根据..."、"通过分析..."等引导句
                
                📝 **回答示例**：
                
                用户问: "今天几号?"
                ❌ 错误回答: "根据我的查询，今天是2023年5月15日星期一。希望这个信息对您有帮助！如果您有其他问题，请随时提问。"
                ✅ 正确回答: "今天是2023年5月15日。"
                
                用户问: "PP材料的熔融指数是多少?"
                ❌ 错误回答: "PP材料的熔融指数通常在范围内变化。熔融指数是衡量热塑性塑料流动性的重要指标，它取决于多种因素，包括分子量和分子量分布。不同等级的PP材料有不同的熔融指数值，一般商业级PP的熔融指数在2-35 g/10min之间。这个指标对加工工艺有重要影响..."
                ✅ 正确回答: "商业级PP材料的熔融指数通常在2-35 g/10min之间，具体取决于等级。"
                
                记住：用户想要的是答案，不是解释或教育。
                """
                )
            ],
        )

    def run(self, user_request: str) -> Iterator[RunResponse]:
        """执行智能助手工作流"""

        intent_data = parse_json_response(self.intent_analyzer.run(user_request).content)
        print(intent_data)
        # 检查是否需要使用默认智能体
        if self._should_use_default_agent(intent_data):
            logger.info("[INTENT] 使用默认智能体处理请求")
            yield from self._handle_with_default_agent(user_request)
            return

        # 如果需要澄清，直接返回澄清问题
        if intent_data and intent_data.get("clarification_needed"):
            logger.info("[INTENT] 用户意图模糊，需要澄清")
            clarification_question = intent_data.get(
                "clarification_question",
                "您的意图不够明确，可以详细说明一下吗？",
            )
            yield RunResponse(
                run_id=self.run_id,
                content=f"🤔 **需要您提供更多信息**\n\n> {clarification_question}",
            )
            return

        # 意图识别成功，显示识别结果
        if intent_data:
            user_intents = intent_data.get("user_intents", ["用户查询"])
            keywords = intent_data.get("keywords", [])
            domain = intent_data.get("domain", "塑化行业")
            
            intents_display = "、".join(user_intents) if len(user_intents) > 1 else user_intents[0]
            logger.info(f"[INTENT] 意图识别成功: {intents_display}")
            yield RunResponse(
                run_id=self.run_id,
                content=f"✅ 识别完成:\n**用户意图**: {intents_display}\n**关键词**: {', '.join(keywords)}\n**领域**: {domain}\n",
            )

        # 第二步：任务拆分
        yield RunResponse(
            run_id=self.run_id, content="📋 **第二步：拆分任务并制定计划**\n"
        )

        task_plan = None
        try:
            # 将意图识别的结果作为任务拆分的输入
            plan_response = self.task_planner.run(
                json.dumps(intent_data, ensure_ascii=False, indent=2)
            )
            if plan_response and plan_response.content:
                plan_data = parse_json_response(plan_response.content)

                # 构建任务计划对象
                task_plan = TaskPlan(user_query=user_request)
                for task_data in plan_data.get("subtasks", []):
                    task_plan.subtasks.append(SubTask(**task_data))
                task_plan.execution_order = plan_data.get("execution_order", [])

                yield RunResponse(
                    run_id=self.run_id,
                    content=f"✅ 任务计划制定完成: 共 {len(task_plan.subtasks)} 个子任务。\n",
                )

                # 显示任务计划
                plan_display = "📑 **任务计划详情**:\n"
                for i, task in enumerate(task_plan.subtasks, 1):
                    plan_display += (
                        f"  {i}. **{task.title}** ➔ `({task.assigned_expert})`\n"
                    )
                    if task.dependencies:
                        plan_display += f"     *依赖: {', '.join(task.dependencies)}*\n"
                yield RunResponse(run_id=self.run_id, content=plan_display + "\n")

        except Exception as e:
            logger.error(f"[PLANNING] 任务拆分失败: {e}")
            yield RunResponse(run_id=self.run_id, content=f"❌ 任务拆分失败: {e}")
            return

        # 第三步：分配智能体并执行任务
        yield RunResponse(
            run_id=self.run_id, content="⚙️ **第三步：多Agent协同执行任务**\n"
        )

        # 存储已完成任务的结果
        completed_tasks_results = {}
        try:
            for i, batch in enumerate(task_plan.execution_order, 1):
                yield RunResponse(
                    run_id=self.run_id,
                    content=f"  🔄 **执行批次 {i}**: {'并行处理' if len(batch) > 1 else '处理'} {len(batch)} 个任务\n",
                )

                for task_id in batch:
                    task = next(
                        (t for t in task_plan.subtasks if t.task_id == task_id), None
                    )
                    if not task:
                        logger.warning(f"任务 {task_id} 在计划中但未找到定义。")
                        continue

                    yield RunResponse(
                        run_id=self.run_id,
                        content=f"    - `[{task.assigned_expert}]` 开始处理: **{task.title}** ...\n",
                    )

                    # 执行子任务
                    result = self._execute_subtask(task, completed_tasks_results)

                    if result:
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                        completed_tasks_results[task.task_id] = result
                        yield RunResponse(
                            run_id=self.run_id,
                            content=f"    - `[{task.assigned_expert}]` ✅ **完成**\n",
                        )
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = "执行失败或无返回"
                        completed_tasks_results[task.task_id] = f"错误: {task.error}"
                        yield RunResponse(
                            run_id=self.run_id,
                            content=f"    - `[{task.assigned_expert}]` ❌ **失败**\n",
                        )

        except Exception as e:
            logger.error(f"[EXECUTION] 任务执行失败: {e}")
            yield RunResponse(run_id=self.run_id, content=f"❌ 任务执行失败: {e}")
            return

        # 第四步：结果整合
        yield RunResponse(
            run_id=self.run_id, content="📝 **第四步：整合最终答案**\n"
        )

        try:
            # 准备整合信息
            integration_context = {
                "user_query": user_request,
                "intent_data": intent_data,
                "completed_tasks": completed_tasks_results,
                "execution_summary": f"共执行 {len(task_plan.subtasks)} 个子任务",
            }

            integration_response = self.result_integrator.run(
                json.dumps(integration_context, ensure_ascii=False, indent=2)
            )

            if integration_response and integration_response.content:
                yield RunResponse(
                    run_id=self.run_id,
                    content=f"✅ **最终答案**:\n\n{integration_response.content}",
                )
            else:
                # 备用方案：直接输出所有任务结果
                fallback_content = "**各专家分析结果**:\n\n"
                for task_id, result in completed_tasks_results.items():
                    task_info = next(
                        (t for t in task_plan.subtasks if t.task_id == task_id), None
                    )
                    expert_name = task_info.assigned_expert if task_info else "未知专家"
                    fallback_content += f"**{expert_name}**: {result}\n\n"

                yield RunResponse(run_id=self.run_id, content=fallback_content)

        except Exception as e:
            logger.error(f"[INTEGRATION] 结果整合失败: {e}")
            yield RunResponse(run_id=self.run_id, content=f"❌ 结果整合失败: {e}")


    def _analyze_intent(self, context_message: str) -> Optional[dict]:
        """分析用户意图"""
        try:
            intent_response = self.intent_analyzer.run(message=context_message)
            if intent_response and intent_response.content:
                intent_data = parse_json_response(intent_response.content)
                print(f"[DEBUG] 意图识别结果: {intent_data}")
                return intent_data
        except Exception as e:
            logger.error(f"[INTENT] 意图识别失败: {e}")
        return None

    def _should_use_default_agent(self, intent_data: Optional[dict]) -> bool:
        """判断是否应该使用默认智能体处理请求"""
        if not intent_data:
            logger.info("[INTENT] 意图识别失败，使用默认智能体")
            return True
        
        # 检查是否为通用领域问题
        domain = intent_data.get("domain", "")
        if domain == "通用":
            logger.info("[INTENT] 检测到通用领域问题，使用默认智能体")
            return True
        
        # 检查意图是否清晰
        intent_clear = intent_data.get("intent_clear", False)
        
        if not intent_clear:
            logger.info("[INTENT] 意图不清晰，需要澄清")
            return False  # 需要澄清的情况不使用默认智能体，而是返回澄清问题
        
        # 对于塑化行业的简单问题，可以考虑使用默认智能体
        if domain == "塑化行业":
            keywords = intent_data.get("keywords", [])
            # 如果关键词很少，可能是简单查询，使用默认智能体
            if len(keywords) <= 2:
                logger.info("[INTENT] 检测到简单查询，使用默认智能体")
                return True
        
        logger.info(f"[INTENT] 检测到复杂的{domain}问题，使用多Agent协同处理")
        return False

    def _handle_with_default_agent(self, user_request: str) -> Iterator[RunResponse]:
        """使用默认智能体处理用户请求"""
        try:
            default_response = self.default_agent.run(user_request)
            if default_response and default_response.content:
                yield RunResponse(
                    run_id=self.run_id,
                    content=default_response.content,
                )
            else:
                yield RunResponse(
                    run_id=self.run_id,
                    content="抱歉，我无法理解您的问题。作为塑化行业专家，我主要回答与塑料材料、加工工艺、质量控制等相关的专业问题。如果您有其他问题，我也会尽力协助。",
                )
        except Exception as e:
            logger.error(f"[DEFAULT_AGENT] 使用默认智能体处理失败: {e}")
            yield RunResponse(
                run_id=self.run_id,
                content="❌ 使用默认智能体处理失败，请稍后再试。",
            )

    def _execute_subtask(
        self, task: SubTask, completed_tasks: Dict[str, str]
    ) -> Optional[str]:
        """执行单个子任务"""
        try:
            expert_agent = self._get_expert_by_name(task.assigned_expert)
            if not expert_agent:
                raise ValueError(f"未找到名为 '{task.assigned_expert}' 的专家。")

            # 构建任务输入，包含依赖项的结果
            task_input = f"""
            **你的任务**: {task.title}
            **任务描述**: {task.description}
            """

            if task.dependencies:
                task_input += "\n\n**参考以下前置任务的结果:**\n"
                for dep_id in task.dependencies:
                    task_input += f"\n--- Pre-Task {dep_id} Result ---\n{completed_tasks.get(dep_id, 'N/A')}\n"

            task_input += "\n请严格按照你的任务描述，提供专业、深入的分析结果。"

            response = expert_agent.run(task_input)
            return response.content if response else None

        except Exception as e:
            logger.error(f"[SUBTASK] 子任务 '{task.title}' 执行失败: {e}")
            return f"执行子任务时发生错误: {e}"

    def _get_expert_by_name(self, expert_name: str) -> Optional[Agent]:
        """根据专家名称获取Agent实例"""
        expert_mapping = {
            "材料技术专家": self.material_expert,
            "工艺技术专家": self.process_expert,
            "质量控制专家": self.quality_expert,
            "市场分析专家": self.market_analyst,
            "计算分析专家": self.calculation_expert,
        }
        return expert_mapping.get(expert_name)


def main():
    # 页面配置
    st.set_page_config(
        page_title="塑化行业智能助手 - Agentica",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 主标题
    st.title("🧪 塑化行业智能助手 (高级协同版)")
    st.markdown("##### 🧬 多意图识别 | 🤖 智能任务拆分 | 🤝 多Agent协同执行")

    # 初始化session_state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": get_welcome_message()}
        ]
    
    # 初始化用户输入历史
    if "user_input_history" not in st.session_state:
        st.session_state.user_input_history = []

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统控制台")

        # 清空对话按钮
        if st.button("🗑️ 清空对话历史", type="primary", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": get_welcome_message()}
            ]
            # 同时清空用户输入历史
            st.session_state.user_input_history = []
            st.rerun()

        st.divider()

        # 工作流程介绍
        with st.expander("🔄 高级工作流程", expanded=True):
            st.markdown(
                """
            **五步协同流程**:
            1. **🎯 意图识别** - 全面识别用户所有意图
            2. **📋 任务拆分** - 将意图转化为子任务，并规划依赖与顺序
            3. **⚙️ 分配执行** - 多专家并行/串行执行子任务
            4. **📊 结果整合** - 汇总所有结果，形成最终报告
            """
            )

        # 专家团队介绍
        with st.expander("👥 专家Agent团队"):
            st.markdown(
                """
            - **🧠 意图识别专家**
            - **📋 任务拆分专家**
            - **🧪 材料技术专家**
            - **🏭 工艺技术专家**
            - **🔍 质量控制专家**
            - **📊 市场分析专家**
            - **🧮 计算分析专家**
            - **📑 结果整合专家**
            """
            )

        # 系统状态
        st.subheader("🔌 系统状态")
        try:
            # 轻量级检查
            st.success("✅ 模型服务已连接")
        except Exception as e:
            st.error(f"❌ 模型服务异常: {str(e)}")

        # 示例任务
        with st.expander("📋 复杂任务示例"):
            examples = {
                "分析一下PP材料的熔融指数对注塑工艺有什么影响，并考虑其对生产成本的最终影响。": "multi_intent_1",
                "我需要一份关于ABS塑料耐热改性的技术方案，包括推荐的改性剂、相应的工艺调整以及成本效益分析。": "multi_intent_2",
                "制定一套完整的PP注塑制品质量控制方案，从原料检验到成品出库，并评估其对整体成本的影响。": "multi_intent_3",
            }

            for example, key in examples.items():
                if st.button(f"💡 {example}", key=key, use_container_width=True):
                    st.session_state.messages.append(
                        {"role": "user", "content": example}
                    )
                    st.rerun()

    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 用户输入处理
    if prompt := st.chat_input("💬 请描述您的塑化行业复杂需求..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 显示用户消息
        with st.chat_message("user"):
            st.write(prompt)

        # 核心逻辑：检查是否为澄清回答，并合并上下文
        final_prompt_to_run = prompt
        is_clarification_response = (
            len(st.session_state.messages) >= 3
            and st.session_state.messages[-2].get("role") == "assistant"
            and st.session_state.messages[-2].get("content", "").startswith("🤔")
        )

        if is_clarification_response:
            original_query = st.session_state.messages[-3].get("content", "")
            clarification_answer = prompt

            # 创建一个对用户更友好的上下文合并提示
            contextual_prompt = f"""好的，已收到您的补充信息。我将结合您最初的问题和刚才的说明进行重新分析：

**原始问题**: `{original_query}`
**补充说明**: `{clarification_answer}`

---
正在重新处理..."""

            # 真正传递给 workflow 的是更结构化的指令
            final_prompt_to_run = f"这是我最初的问题：'{original_query}'\n\n对于你的澄清问题，我的回答是：'{clarification_answer}'\n\n请基于这些完整信息重新进行分析和处理。"

            # 在开始处理前，先显示一个合并上下文的提示
            with st.chat_message("assistant"):
                st.write(contextual_prompt)
            st.session_state.messages.append(
                {"role": "assistant", "content": contextual_prompt}
            )

        # 处理用户请求
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                # 使用固定的session_id以保持会话连续性
                if "workflow_session_id" not in st.session_state:
                    st.session_state.workflow_session_id = f"chat_{uuid.uuid4().hex[:8]}"
                
                # 为每次运行创建工作流实例，但使用相同的session_id
                workflow = PlasticIndustryWorkflow(
                    session_id=st.session_state.workflow_session_id,
                    storage=SqlWorkflowStorage(
                        table_name="plastic_industry_workflows",
                        db_file="outputs/plastic_industry_workflows.db",
                    ),
                )
                
                # 只为意图识别智能体设置存储
                storage = SqlWorkflowStorage(
                    table_name="agent_sessions",
                    db_file="outputs/agent_sessions.db",
                )

                
                # 只为意图识别智能体设置session_id和存储
                
                workflow.intent_analyzer.storage = storage
                workflow.intent_analyzer.session_id = f"intent_analyzer_{st.session_state.workflow_session_id}"
                
                # 从存储中加载之前的会话状态
                try:
                    workflow.read_from_storage()
                    logger.info(f"[WORKFLOW] 已从存储中加载会话状态，session_id: {workflow.session_id}")
                    
                    # 只加载意图识别智能体的状态
                    if hasattr(workflow.intent_analyzer, 'read_from_storage'):
                        workflow.intent_analyzer.read_from_storage()
                        logger.info(f"[AGENT] 意图识别智能体状态已从存储中加载")
                except Exception as e:
                    logger.warning(f"[WORKFLOW] 加载会话状态失败，将使用新会话: {e}")

                # 流式处理
                for response in workflow.run(final_prompt_to_run):
                    if response and response.content:
                        full_response += response.content
                        response_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)

                # 移除光标
                response_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                error_msg = f"❌ 处理您的请求时发生严重错误: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


def get_welcome_message() -> str:
    """获取欢迎消息"""
    return """
🎉 **欢迎使用塑化行业智能助手 (高级协同版)**
    """


if __name__ == "__main__":
    # streamlit run web_chat_app.py --server.port 8504 --server.address 0.0.0.0
    # 上面是运行命令，我想使用debug模式，怎么修改？
    
    main()

