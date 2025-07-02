#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试智能任务处理器的意图分析功能
去掉增强版意图识别器，直接使用AI进行意图分析
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentica.storage.workflow.sqlite import SqlWorkflowStorage
from web_chat_app import SmartTaskDecomposer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_intent_analysis.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_intent_analysis():
    """测试意图分析功能"""
    print("=" * 80)
    print("🧠 智能任务处理器 - 意图分析测试")
    print("=" * 80)
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    # 测试用例
    test_cases = [
        {
            "name": "简单计算任务",
            "query": "计算 25 * 36 + 180",
            "expected_complexity": "simple"
        },
        {
            "name": "天气查询任务", 
            "query": "今天北京的天气怎么样？",
            "expected_complexity": "simple"
        },
        {
            "name": "随机数生成",
            "query": "生成5个1-100的随机数",
            "expected_complexity": "simple"
        },
        {
            "name": "复杂研究任务",
            "query": "分析人工智能在医疗领域的应用现状，包括技术发展趋势和市场前景",
            "expected_complexity": "complex"
        },
        {
            "name": "计划制定任务",
            "query": "制定一个详细的一个月健身计划，包括有氧运动、力量训练和饮食建议",
            "expected_complexity": "complex"
        }
    ]
    
    # 创建智能任务处理器
    decomposer = SmartTaskDecomposer(
        session_id=f"test-intent-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        storage=SqlWorkflowStorage(
            table_name="test_intent_workflows",
            db_file="outputs/test_intent_workflows.db",
        ),
    )
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'-' * 60}")
        print(f"📝 测试用例 {i}/{total_count}: {test_case['name']}")
        print(f"🎯 用户请求: {test_case['query']}")
        print(f"🔮 预期复杂度: {test_case['expected_complexity']}")
        print(f"{'-' * 60}")
        
        try:
            # 执行任务处理
            response_content = ""
            step_count = 0
            
            for response in decomposer.run(test_case['query']):
                step_count += 1
                if response.content:
                    response_content += response.content
                    print(f"📄 步骤 {step_count}: {response.content[:100]}...")
            
            # 评估结果
            if response_content:
                print(f"✅ 任务处理成功")
                print(f"📊 处理步骤数: {step_count}")
                print(f"📝 响应长度: {len(response_content)} 字符")
                success_count += 1
            else:
                print(f"❌ 任务处理失败：无响应内容")
                
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            logger.exception(f"测试用例 '{test_case['name']}' 执行失败")
    
    # 输出测试总结
    print(f"\n" + "=" * 80)
    print(f"📊 测试总结")
    print(f"=" * 80)
    print(f"✅ 成功: {success_count}/{total_count}")
    print(f"❌ 失败: {total_count - success_count}/{total_count}")
    print(f"📈 成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print(f"🎉 所有测试用例都通过了！意图分析功能正常工作")
    else:
        print(f"⚠️  有 {total_count - success_count} 个测试用例失败，请检查日志文件")
    
    return success_count == total_count

if __name__ == "__main__":
    try:
        success = test_intent_analysis()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试执行出现意外错误: {str(e)}")
        logger.exception("测试执行失败")
        sys.exit(1) 