# -*- coding: utf-8 -*-
"""
自定义工具使用示例
演示如何在 Agentica Agent 中使用自定义工具
"""
import os
import sys

# 添加项目根目录到 Python 路径，确保能正确导入 agentica 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agentica import Agent, OpenAIChat
from agentica.tools.random_tool import RandomTool
from agentica.tools.text_processor_tool import TextProcessorTool


def run_custom_tools_demo():
    """演示自定义工具的使用"""
    
    print("=" * 60)
    print("🚀 Agentica 自定义工具使用示例")
    print("=" * 60)
    
    # 检查是否设置了 OpenAI API 密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  注意：未检测到 OPENAI_API_KEY 环境变量")
        print("🔧 这个示例将使用本地工具测试，不会调用 AI 模型")
        print("\n如果要使用完整功能，请设置：")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\n" + "=" * 60)
        
        # 本地工具测试
        print("\n🧪 本地工具功能测试：")
        test_tools_locally()
        return
    
    # 创建带有自定义工具的 Agent
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),  # 使用更便宜的模型
        tools=[
            RandomTool(),           # 随机数工具
            TextProcessorTool(),    # 文本处理工具
        ],
        show_tool_calls=True,       # 显示工具调用过程
        markdown=True,              # 支持 Markdown 输出
    )
    
    print("\n🤖 Agent 已创建，包含以下自定义工具：")
    print("   • RandomTool - 随机数生成工具")
    print("   • TextProcessorTool - 文本处理工具")
    
    # 测试用例
    test_cases = [
        {
            "description": "随机数生成测试",
            "query": "请生成一个1到100之间的随机数，然后生成5个1到10之间的随机数列表"
        },
        {
            "description": "文本处理测试", 
            "query": """请分析这段文本的统计信息，并提取其中的邮箱和网址：
            "欢迎访问我们的网站 https://example.com，如有疑问请联系 support@company.com 或 info@help.org。
            这是一个包含多种信息的测试文本，用于演示文本处理功能。\""""
        },
        {
            "description": "文本清理和转换测试",
            "query": "请将文本 'Hello World!!! 123 Test???' 进行清理（移除数字和标点），然后转换为标题格式"
        },
        {
            "description": "组合功能测试",
            "query": "请从这些选项中随机选择一个：['Python', 'JavaScript', 'Go', 'Rust']，然后分析选中结果的文本统计信息"
        }
    ]
    
    print("\n" + "=" * 60)
    print("🎯 开始测试用例")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试 {i}: {test_case['description']}")
        print("-" * 40)
        print(f"🗨️  问题: {test_case['query']}")
        print("\n🤖 Agent 回复:")
        
        try:
            agent.print_response(test_case['query'])
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
        
        print("\n" + "-" * 40)
        print("✅ 测试完成")
        
        # 询问是否继续
        if i < len(test_cases):
            user_input = input(f"\n按 Enter 继续下一个测试，或输入 'q' 退出: ").strip()
            if user_input.lower() == 'q':
                break


def test_tools_locally():
    """本地测试工具功能（不需要 API 密钥）"""
    
    # 测试 RandomTool 随机数生成工具
    print("\n🎲 RandomTool 测试:")
    random_tool = RandomTool()
    
    print("1. 生成随机数 (1-100):")
    result = random_tool.generate_random_number(1, 100)
    print(f"   结果: {result}")
    
    print("\n2. 生成随机数列表 (5个, 1-10):")
    result = random_tool.generate_random_list(5, 1, 10)
    print(f"   结果: {result}")
    
    print("\n3. 随机选择:")
    result = random_tool.generate_random_choice(["苹果", "香蕉", "橙子", "葡萄"])
    print(f"   结果: {result}")
    
    # 测试 TextProcessorTool 文本处理工具
    print("\n\n📝 TextProcessorTool 测试:")
    text_tool = TextProcessorTool()
    
    # 中英文混合测试文本
    test_text = "Hello World! 请联系我们：info@example.com。访问 https://example.com 获取更多信息。这是一个包含123个数字的测试。"
    
    print("1. 文本统计:")
    result = text_tool.count_words(test_text)
    print(f"   结果: {result}")
    
    print("\n2. 提取邮箱:")
    result = text_tool.extract_emails(test_text)
    print(f"   结果: {result}")
    
    print("\n3. 提取URL:")
    result = text_tool.extract_urls(test_text)
    print(f"   结果: {result}")
    
    print("\n4. 文本清理 (移除数字和标点):")
    result = text_tool.clean_text(test_text, remove_numbers=True, remove_punctuation=True)
    print(f"   结果: {result}")
    
    print("\n5. 大小写转换 (标题格式):")
    result = text_tool.transform_case("hello world test", "title")
    print(f"   结果: {result}")
    
    print("\n6. 生成哈希 (SHA256):")
    result = text_tool.generate_hash("hello world", "sha256")
    print(f"   结果: {result}")


def show_instructions():
    """显示使用说明"""
    print("\n" + "=" * 60)
    print("📚 如何创建自定义工具")
    print("=" * 60)
    
    instructions = """
    
🔧 创建自定义工具的步骤：

1. 继承 Tool 基类
   from agentica.tools.base import Tool

2. 在 __init__ 中注册函数
   def __init__(self):
       super().__init__(name="my_tool")
       self.register(self.my_function)

3. 编写工具函数
   def my_function(self, param: str) -> str:
       \"\"\"函数说明
       
       Args:
           param (str): 参数说明
           
       Returns:
           str: 返回 JSON 字符串
       \"\"\"
       # 实现功能
       return json.dumps({"result": "success"})

4. 在 Agent 中使用
   agent = Agent(
       model=OpenAIChat(),
       tools=[MyTool()],
       show_tool_calls=True
   )

📍 关键要点：
• 函数必须有类型注解（AI 需要这些信息）
• 文档字符串很重要（AI 理解功能的依据）
• 返回 JSON 字符串（便于 AI 处理）
• 调用 self.register() 注册函数

📂 工具文件位置：
• 放在 agentica/tools/ 目录下
• 文件名格式：your_tool_name_tool.py
• 类名格式：YourToolNameTool

📦 添加到项目：
• 在 agentica/__init__.py 中导入
• 在 agentica/cli.py 的 TOOL_MAP 中添加映射

🎯 本示例创建的工具：
• RandomTool: 随机数生成功能
• TextProcessorTool: 文本处理功能

这些工具现在已经集成到 Agentica 项目中，可以在 CLI 和代码中使用！
    """
    
    print(instructions)


if __name__ == "__main__":
    # 显示说明
    show_instructions()
    
    # 运行演示
    try:
        run_custom_tools_demo()
    except KeyboardInterrupt:
        print("\n\n👋 演示已中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎉 自定义工具演示完成！")
    print("📖 查看源码了解更多实现细节")
    print("=" * 60) 