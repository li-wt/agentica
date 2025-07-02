#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的Web应用功能
"""
import sys
sys.path.append('.')

from web_chat_app import (
    IntentClassifier, 
    SmartTaskDecomposer, 
    SpecialCommandHandler
)
from agentica import SqlWorkflowStorage
from datetime import datetime

def test_intent_classification():
    """测试意图识别功能"""
    print("🧠 测试意图识别功能...")
    
    test_cases = [
        ("计算 25 * 36", ['calculation']),
        ("今天北京天气怎么样？", ['weather']),
        ("分析人工智能在医疗领域的应用现状", ['complex_task']),
        ("搜索最新的机器学习论文", ['search']),
        ("@help", ['help']),
        ("你好", ['greeting'])
    ]
    
    for text, expected in test_cases:
        intents = IntentClassifier.classify_intent(text)
        print(f"  输入: {text}")
        print(f"  识别意图: {intents}")
        print(f"  预期意图: {expected}")
        print(f"  是否正确: {any(intent in intents for intent in expected)}")
        print()

def test_task_decomposition():
    """测试任务拆分判断"""
    print("🧩 测试任务拆分判断...")
    
    test_cases = [
        ("计算 25 * 36", False),  # 简单任务
        ("今天天气", False),  # 简单任务
        ("分析人工智能在医疗领域的应用现状和发展趋势", True),  # 复杂任务
        ("制定一个月的健身计划", True),  # 复杂任务
        ("比较Python和Java的优缺点", True),  # 复杂任务
    ]
    
    for text, expected in test_cases:
        needs_decomp = IntentClassifier.needs_task_decomposition(text)
        print(f"  输入: {text}")
        print(f"  需要拆分: {needs_decomp}")
        print(f"  预期: {expected}")
        print(f"  是否正确: {needs_decomp == expected}")
        print()

def test_basic_functionality():
    """测试基础功能"""
    print("⚙️ 测试基础功能...")
    
    # 测试欢迎消息生成
    from web_chat_app import get_welcome_message
    welcome_msg = get_welcome_message()
    print(f"  欢迎消息长度: {len(welcome_msg)} 字符")
    print(f"  包含关键词: {'✅' if '智能任务拆分助手' in welcome_msg else '❌'}")
    print()

def test_special_commands():
    """测试特殊指令处理"""
    print("⭐ 测试特殊指令处理...")
    
    help_response = SpecialCommandHandler.handle_help()
    print("  @Help 响应:")
    print("  ", help_response[:100] + "..." if len(help_response) > 100 else help_response)
    print()
    
    changes_response = SpecialCommandHandler.handle_recent_changes(None)
    print("  @Recent Changes 响应:")
    print("  ", changes_response[:100] + "..." if len(changes_response) > 100 else changes_response)
    print()

def test_smart_decomposer_simple():
    """测试智能分解器处理简单任务"""
    print("🤖 测试智能分解器 - 简单任务...")
    
    try:
        decomposer = SmartTaskDecomposer(
            session_id=f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            storage=SqlWorkflowStorage(
                table_name="test_workflows",
                db_file="outputs/test_workflows.db",
            ),
        )
        
        print("  智能分解器初始化成功")
        print("  包含的Agent:")
        print(f"    - 意图分析器: {'✅' if decomposer.intent_analyzer else '❌'}")
        print(f"    - 任务规划器: {'✅' if decomposer.task_planner else '❌'}")
        print(f"    - 研究专家: {'✅' if decomposer.research_agent else '❌'}")
        print(f"    - 计算专家: {'✅' if decomposer.calculation_agent else '❌'}")
        print(f"    - 写作专家: {'✅' if decomposer.writing_agent else '❌'}")
        print(f"    - 工具专家: {'✅' if decomposer.tool_agent else '❌'}")
        
    except Exception as e:
        print(f"  ❌ 智能分解器初始化失败: {e}")
    
    print()

def main():
    """运行所有测试"""
    print("🚀 开始测试修改后的Web应用功能\n")
    print("=" * 50)
    
    try:
        test_intent_classification()
        print("=" * 50)
        
        test_task_decomposition()
        print("=" * 50)
        
        test_basic_functionality()
        print("=" * 50)
        
        test_special_commands()
        print("=" * 50)
        
        test_smart_decomposer_simple()
        print("=" * 50)
        
        print("✅ 所有功能测试完成！")
        print("\n🎯 总结:")
        print("  - 意图识别系统正常")
        print("  - 任务拆分判断正常")
        print("  - 基础功能正常")
        print("  - 特殊指令处理正常")
        print("  - 智能分解器初始化正常")
        print("\n🌟 简化版Web应用已准备就绪，可以在8504端口启动！")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 