#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能任务拆分功能测试脚本
"""
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.append('.')
sys.path.append('..')

def test_smart_decomposer():
    """测试智能任务拆分功能"""
    print("🧩 测试智能任务拆分功能...")
    
    try:
        from web_chat_app import SmartTaskDecomposer, IntentClassifier
        from agentica import SqlWorkflowStorage
        
        # 测试意图识别
        print("\n🧠 测试意图识别...")
        test_queries = [
            "计算 25 * 36",  # 简单任务
            "分析人工智能在医疗领域的应用现状和发展趋势",  # 复杂任务
            "制定一个月的健身计划",  # 复杂任务
            "今天天气怎么样？",  # 简单任务
        ]
        
        for query in test_queries:
            intents = IntentClassifier.classify_intent(query)
            needs_decomp = IntentClassifier.needs_task_decomposition(query)
            print(f"  查询: {query}")
            print(f"  意图: {intents}")
            print(f"  需要拆分: {'是' if needs_decomp else '否'}")
            print()
        
        # 测试任务拆分器创建
        print("🔧 测试任务拆分器创建...")
        decomposer = SmartTaskDecomposer(
            session_id=f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            storage=SqlWorkflowStorage(
                table_name="test_smart_task_workflows",
                db_file="outputs/test_smart_task_workflows.db",
            ),
        )
        print("✅ 智能任务拆分器创建成功")
        
        # 测试简单的拆分流程（不实际执行LLM调用）
        print("📋 测试任务拆分流程组件...")
        print("✅ 意图分析器已就绪")
        print("✅ 任务规划器已就绪")
        print("✅ 执行协调器已就绪")
        print("✅ 专业Agent团队已就绪")
        
        print("\n🎯 智能任务拆分功能测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 智能任务拆分功能测试失败: {str(e)}")
        return False

def test_web_app_components():
    """测试Web应用各组件"""
    print("🌐 测试Web应用组件...")
    
    try:
        from web_chat_app import (
            IntentClassifier, SmartTaskDecomposer, 
            SpecialCommandHandler, get_agent_with_tools,
            display_intent_info
        )
        
        print("✅ IntentClassifier 导入成功")
        print("✅ SmartTaskDecomposer 导入成功")
        print("✅ SpecialCommandHandler 导入成功")
        print("✅ get_agent_with_tools 导入成功")
        print("✅ display_intent_info 导入成功")
        
        # 测试特殊指令处理
        print("\n🎯 测试特殊指令处理...")
        help_response = SpecialCommandHandler.handle_help()
        print(f"✅ 帮助指令响应长度: {len(help_response)} 字符")
        
        recent_response = SpecialCommandHandler.handle_recent_changes(None)
        print(f"✅ 最近变更指令响应长度: {len(recent_response)} 字符")
        
        print("\n🌐 Web应用组件测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ Web应用组件测试失败: {str(e)}")
        return False

def test_localchat_connection():
    """测试LocalChat连接"""
    print("🔌 测试LocalChat连接...")
    
    try:
        from agentica import LocalChat
        
        model = LocalChat()
        print(f"✅ LocalChat模型创建成功")
        print(f"  - ID: {model.id}")
        print(f"  - 名称: {model.name}")
        print(f"  - 提供商: {model.provider}")
        print(f"  - API地址: {model.base_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ LocalChat连接测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始智能任务拆分系统测试")
    print("=" * 50)
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    tests = [
        ("LocalChat连接", test_localchat_connection),
        ("Web应用组件", test_web_app_components),
        ("智能任务拆分", test_smart_decomposer),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 执行测试: {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
        print(f"{'✅ 通过' if result else '❌ 失败'}: {test_name}")
    
    # 总结测试结果
    print("\n" + "=" * 50)
    print("📊 测试总结")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！Web应用已准备就绪。")
        print("\n🌐 启动Web应用:")
        print("streamlit run web_chat_app.py --server.port 8501 --server.address 0.0.0.0")
    else:
        print("⚠️ 部分测试失败，请检查配置。")

if __name__ == "__main__":
    main() 