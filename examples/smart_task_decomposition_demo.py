# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 智能任务拆分示例 - 根据用户意图动态分解任务
"""

import sys
from textwrap import dedent
from typing import Optional, Dict, List, Iterator
from pydantic import BaseModel, Field
from loguru import logger
import json

sys.path.append('..')
from agentica import Agent, LocalChat
from agentica.workflow import Workflow
from agentica import RunResponse, RunEvent, SqlWorkflowStorage, pprint_run_response
from agentica.tools.search_serper_tool import SearchSerperTool
from agentica.tools.wikipedia_tool import WikipediaTool
from agentica.tools.calculator_tool import CalculatorTool
from agentica.tools.run_python_code_tool import RunPythonCodeTool


class TaskStep(BaseModel):
    step_id: str = Field(..., description="步骤唯一标识")
    step_name: str = Field(..., description="步骤名称")
    description: str = Field(..., description="步骤详细描述")
    required_tools: List[str] = Field(..., description="需要的工具列表")
    depends_on: List[str] = Field(default_factory=list, description="依赖的前置步骤ID")
    estimated_time: int = Field(..., description="预估时间（分钟）")
    priority: int = Field(..., description="优先级 1-10")


class TaskPlan(BaseModel):
    task_type: str = Field(..., description="任务类型")
    complexity: str = Field(..., description="复杂度：simple/medium/complex")
    steps: List[TaskStep] = Field(..., description="任务步骤列表")
    total_estimated_time: int = Field(..., description="总预估时间")
    success_criteria: str = Field(..., description="成功标准")


class SmartTaskDecomposer(Workflow):
    """智能任务分解器 - 根据用户意图动态分解任务"""
    
    description: str = "智能分析用户需求，动态生成执行计划并协调多个专业Agent完成复杂任务"
    
    # 声明Agent字段
    intent_analyzer: Optional[Agent] = None
    task_planner: Optional[Agent] = None
    execution_coordinator: Optional[Agent] = None
    research_agent: Optional[Agent] = None
    calculation_agent: Optional[Agent] = None
    writing_agent: Optional[Agent] = None

    def __init__(self, **data):
        super().__init__(**data)
        
        # 在初始化时创建Agent，避免类级别的Agent定义
        self.intent_analyzer = Agent(
            model=LocalChat(),
            session_id=self.session_id,
            instructions=[
                dedent("""
                你是一个专业的意图分析专家。分析用户的请求，识别：
                1. 核心意图和目标
                2. 任务类型和复杂度
                3. 需要的专业领域知识
                4. 可能的执行路径
                5. 成功标准
                
                输出格式必须是JSON：
                {
                    "core_intent": "核心意图描述",
                    "task_type": "任务类型",
                    "complexity": "simple/medium/complex",
                    "domains": ["领域1", "领域2"],
                    "success_criteria": "成功标准",
                    "key_challenges": ["挑战1", "挑战2"]
                }
                """),
            ],
        )

        self.task_planner = Agent(
            model=LocalChat(),
            session_id=self.session_id,
            tools=[SearchSerperTool(), WikipediaTool()],
            instructions=[
                dedent("""
                你是一个专业的任务规划专家。根据意图分析结果，制定详细的执行计划。
                
                规划原则：
                1. 将复杂任务分解为可执行的子任务
                2. 识别任务间的依赖关系
                3. 选择合适的工具和方法
                4. 考虑执行顺序和并行可能性
                5. 设定检查点和调整机制
                
                如果需要外部信息来制定更好的计划，使用搜索工具获取相关信息。
                
                输出必须严格按照TaskPlan格式的JSON。
                """),
            ],
            response_model=TaskPlan,
        )

        self.execution_coordinator = Agent(
            model=LocalChat(),
            session_id=self.session_id,
            instructions=[
                dedent("""
                你是执行协调器，负责：
                1. 按计划执行任务步骤
                2. 监控执行进度和质量
                3. 在需要时调整计划
                4. 整合各步骤结果
                5. 确保最终目标达成
                
                执行过程中要实时反馈进度和遇到的问题。
                """),
            ],
        )

        self.research_agent = Agent(
            model=LocalChat(),
            session_id=self.session_id,
            tools=[SearchSerperTool(), WikipediaTool()],
            instructions=["你是研究专家，负责信息收集、分析和整理。"],
        )

        self.calculation_agent = Agent(
            model=LocalChat(),
            session_id=self.session_id,
            tools=[CalculatorTool(), RunPythonCodeTool()],
            instructions=["你是计算专家，负责数据分析、计算和可视化。"],
        )

        self.writing_agent = Agent(
            model=LocalChat(),
            session_id=self.session_id,
            instructions=["你是写作专家，负责内容创作、编辑和优化。"],
        )

    def run(self, user_request: str) -> Iterator[RunResponse]:
        logger.info(f"收到用户请求: {user_request}")
        
        # 第一步：意图分析
        logger.info("🧠 开始意图分析...")
        intent_response = self.intent_analyzer.run(user_request)
        if not intent_response.content:
            yield RunResponse(run_id=self.run_id, content="意图分析失败")
            return
            
        try:
            intent_data = json.loads(intent_response.content)
            logger.info(f"意图分析结果: {intent_data}")
        except json.JSONDecodeError:
            yield RunResponse(run_id=self.run_id, content="意图分析结果解析失败")
            return

        # 第二步：任务规划
        logger.info("📋 开始任务规划...")
        planning_input = f"""
        用户请求: {user_request}
        意图分析: {json.dumps(intent_data, ensure_ascii=False, indent=2)}
        
        请制定详细的执行计划。
        """
        
        plan_response = self.task_planner.run(planning_input)
        if not plan_response.content or not isinstance(plan_response.content, TaskPlan):
            yield RunResponse(run_id=self.run_id, content="任务规划失败")
            return
            
        task_plan: TaskPlan = plan_response.content
        logger.info(f"任务规划完成: {task_plan.task_type}, 共{len(task_plan.steps)}个步骤")

        # 第三步：动态执行
        logger.info("🚀 开始执行任务...")
        yield RunResponse(
            run_id=self.run_id, 
            content=f"## 任务执行计划\n\n**任务类型**: {task_plan.task_type}\n**复杂度**: {task_plan.complexity}\n**预估时间**: {task_plan.total_estimated_time}分钟\n\n### 执行步骤:\n"
        )
        
        # 按依赖关系执行步骤
        completed_steps = set()
        step_results = {}
        
        for step in task_plan.steps:
            # 检查依赖是否满足
            if not all(dep in completed_steps for dep in step.depends_on):
                continue
                
            logger.info(f"执行步骤: {step.step_name}")
            yield RunResponse(
                run_id=self.run_id,
                content=f"\n### 🔄 执行步骤: {step.step_name}\n{step.description}\n"
            )
            
            # 根据步骤类型选择合适的Agent
            agent = self._select_agent_for_step(step)
            if agent:
                step_input = self._prepare_step_input(step, step_results, user_request)
                step_result = agent.run(step_input)
                
                if step_result.content:
                    step_results[step.step_id] = step_result.content
                    completed_steps.add(step.step_id)
                    
                    yield RunResponse(
                        run_id=self.run_id,
                        content=f"**结果**: {step_result.content}\n"
                    )
                else:
                    yield RunResponse(
                        run_id=self.run_id,
                        content=f"⚠️ 步骤执行失败: {step.step_name}\n"
                    )

        # 第四步：结果整合
        logger.info("📊 整合最终结果...")
        final_integration_input = f"""
        原始用户请求: {user_request}
        执行计划: {task_plan.model_dump_json(indent=2)}
        各步骤结果: {json.dumps(step_results, ensure_ascii=False, indent=2)}
        
        请整合所有结果，生成最终的完整回答。
        """
        
        final_result = self.execution_coordinator.run(final_integration_input)
        if final_result.content:
            yield RunResponse(
                run_id=self.run_id,
                content=f"\n## 🎯 最终结果\n\n{final_result.content}"
            )

    def _select_agent_for_step(self, step: TaskStep) -> Optional[Agent]:
        """根据步骤需求选择合适的Agent"""
        if "search" in step.required_tools or "research" in step.step_name.lower():
            return self.research_agent
        elif "calculate" in step.required_tools or "python" in step.required_tools:
            return self.calculation_agent
        elif "write" in step.step_name.lower() or "content" in step.step_name.lower():
            return self.writing_agent
        else:
            return self.execution_coordinator

    def _prepare_step_input(self, step: TaskStep, previous_results: Dict, original_request: str) -> str:
        """为步骤准备输入"""
        input_parts = [
            f"原始请求: {original_request}",
            f"当前步骤: {step.step_name}",
            f"步骤描述: {step.description}",
        ]
        
        # 添加依赖步骤的结果
        if step.depends_on:
            input_parts.append("前置步骤结果:")
            for dep_id in step.depends_on:
                if dep_id in previous_results:
                    input_parts.append(f"- {dep_id}: {previous_results[dep_id]}")
        
        return "\n".join(input_parts)


# 使用示例
if __name__ == "__main__":
    # 测试不同类型的复杂请求
    test_requests = [
        "我想了解人工智能在医疗领域的应用现状，分析其优势和挑战，并预测未来发展趋势",
        "帮我制定一个月的健身计划，包括饮食建议，并计算预期的卡路里消耗",
        "分析比特币过去一年的价格走势，找出影响因素，并给出投资建议",
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n{'='*60}")
        print(f"测试案例 {i}: {request}")
        print('='*60)
        
        smart_decomposer = SmartTaskDecomposer(
            session_id=f"smart-task-{i}",
            storage=SqlWorkflowStorage(
                table_name="smart_task_workflows",
                db_file="tmp/smart_task_workflows.db",
            ),
        )
        
        # 执行智能任务分解
        result_stream = smart_decomposer.run(request)
        pprint_run_response(result_stream, markdown=True) 