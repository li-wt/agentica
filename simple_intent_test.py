#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的意图分析测试 - 演示新的日志打印功能
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_intent_flow():
    """测试简化版意图分析流程"""
    print("=" * 80)
    print("🧠 简化版意图分析测试")
    print("=" * 80)
    
    # 模拟用户请求
    test_queries = [
        "计算 25 * 36 + 180",
        "今天北京的天气怎么样？",
        "生成5个1-100的随机数",
        "分析人工智能在医疗领域的应用现状"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-' * 60}")
        print(f"📝 测试用例 {i}: {query}")
        print(f"{'-' * 60}")
        
        # 模拟日志输出（就像真实的SmartTaskDecomposer会输出的）
        print(f"[TASK] 开始处理用户请求: {query}")
        print(f"[INTENT] 启动意图分析器分析用户请求")
        
        # 模拟意图分析过程
        if "计算" in query or "算" in query or any(op in query for op in ['+', '-', '*', '/']):
            intent_result = {
                "core_intent": "数学计算",
                "task_type": "计算任务", 
                "complexity": "simple",
                "domains": ["数学"],
                "success_criteria": "得到正确的计算结果",
                "key_challenges": []
            }
        elif "天气" in query:
            intent_result = {
                "core_intent": "信息查询",
                "task_type": "天气查询",
                "complexity": "simple", 
                "domains": ["气象"],
                "success_criteria": "获取准确的天气信息",
                "key_challenges": []
            }
        elif "随机" in query or "生成" in query:
            intent_result = {
                "core_intent": "数据生成",
                "task_type": "随机数生成",
                "complexity": "simple",
                "domains": ["工具"],
                "success_criteria": "生成符合要求的随机数",
                "key_challenges": []
            }
        else:
            intent_result = {
                "core_intent": "信息分析",
                "task_type": "复杂研究任务",
                "complexity": "complex",
                "domains": ["技术", "医疗"],
                "success_criteria": "提供全面深入的分析报告",
                "key_challenges": ["信息收集", "深度分析", "趋势预测"]
            }
        
        print(f"[INTENT] 意图分析原始响应: {intent_result}")
        print(f"[INTENT] 解析后的意图数据: {intent_result}")
        
        # 输出意图分析结果
        core_intent = intent_result['core_intent']
        task_type = intent_result['task_type']
        complexity = intent_result['complexity']
        domains = intent_result['domains']
        success_criteria = intent_result['success_criteria']
        key_challenges = intent_result['key_challenges']
        
        print(f"✅ **意图分析结果**")
        print(f"- 🎯 核心意图: {core_intent}")
        print(f"- 📝 任务类型: {task_type}")
        print(f"- ⚡ 复杂度: {complexity}")
        print(f"- 🔍 涉及领域: {', '.join(domains) if domains else '通用'}")
        print(f"- 🎖️ 成功标准: {success_criteria}")
        print(f"- 🚧 关键挑战: {', '.join(key_challenges) if key_challenges else '无'}")
        
        print(f"[DECISION] 根据复杂度 '{complexity}' 决定处理方式")
        
        # 根据复杂度决定处理方式
        if complexity in ['simple', 'medium']:
            print(f"[SIMPLE] 识别为{complexity}任务，选择直接处理模式")
            print(f"⚡ **第二步：直接处理（{complexity}任务）**")
            
            # Agent选择逻辑演示
            if '计算' in core_intent or '数学' in core_intent:
                selected_agent = "计算专家Agent"
            elif '查询' in core_intent or '搜索' in core_intent:
                selected_agent = "研究专家Agent"
            elif '生成' in core_intent or '工具' in domains:
                selected_agent = "工具专家Agent"
            else:
                selected_agent = "执行协调器Agent"
            
            print(f"[SIMPLE] 选择Agent: {selected_agent}")
            print(f"[SIMPLE] 处理成功，结果长度: 150 字符")
            print(f"## 🎯 **最终结果**")
            print(f"[模拟结果] 针对'{query}'的处理结果...")
            print(f"---")
            print(f"*任务处理完成，使用直接处理模式*")
            
        else:
            print(f"[COMPLEX] 识别为复杂任务，启动任务拆分模式")
            print(f"📋 **第二步：任务规划**")
            print(f"[PLAN] 发送任务规划请求给task_planner")
            print(f"[PLAN] 任务计划制定成功，共3个步骤，预估15分钟")
            print(f"📋 **第三步：动态执行**")
            print(f"[EXECUTE] 开始执行任务计划，共3个步骤")
            print(f"[EXECUTE] 开始执行步骤: step1 - 信息收集")
            print(f"[EXECUTE] 为步骤step1选择Agent: 研究专家Agent")
            print(f"[EXECUTE] 步骤step1执行成功，结果长度: 500 字符")
            print(f"📊 **第四步：结果整合**")
            print(f"[INTEGRATE] 开始整合最终结果，已完成3个步骤")
            print(f"[INTEGRATE] 结果整合成功，最终结果长度: 1200 字符")
            print(f"## 🎯 **最终整合结果**")
            print(f"[模拟结果] 针对'{query}'的详细分析报告...")
            print(f"---")
            print(f"*任务执行完成，共完成 3 个步骤*")
    
    print(f"\n" + "=" * 80)
    print(f"🎉 意图分析流程测试完成！")
    print(f"✅ 新的日志系统能够清晰展示每一步的运行逻辑")
    print(f"📊 意图分析结果详细展示，便于观察和调试")
    print(f"⚡ 简单任务直接处理，复杂任务智能拆分")
    print(f"=" * 80)

if __name__ == "__main__":
    test_simple_intent_flow() 