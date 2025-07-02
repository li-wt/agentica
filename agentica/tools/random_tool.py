# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 随机数生成工具
"""
import json
import random
import time
from typing import List
from agentica.tools.base import Tool
from agentica.utils.log import logger


class RandomTool(Tool):
    """随机数生成工具
    
    提供多种随机数生成功能，包括单个随机数、随机数列表和随机选择等。
    支持自定义范围和数量，并返回详细的统计信息。
    """
    
    def __init__(self):
        """初始化随机数工具"""
        super().__init__(name="random_tool")
        # 注册工具函数，让AI能够调用这些功能
        self.register(self.generate_random_number)
        self.register(self.generate_random_list)
        self.register(self.generate_random_choice)

    def generate_random_number(self, min_val: int = 1, max_val: int = 100) -> str:
        """生成指定范围内的随机数

        Args:
            min_val (int): 最小值，默认为1
            max_val (int): 最大值，默认为100

        Returns:
            str: 包含随机数和元数据的JSON字符串

        Example:
            from agentica.tools.random_tool import RandomTool
            tool = RandomTool()
            result = tool.generate_random_number(1, 50)
            print(result)
        """
        # 检查参数有效性
        if min_val > max_val:
            logger.error(f"无效范围: min_val ({min_val}) > max_val ({max_val})")
            return json.dumps({
                "operation": "generate_random_number", 
                "error": "最小值不能大于最大值"
            })

        try:
            # 生成随机数
            random_num = random.randint(min_val, max_val)
            logger.info(f"生成随机数: {random_num} (范围: {min_val}-{max_val})")
            
            # 返回包含详细信息的JSON
            return json.dumps({
                "operation": "generate_random_number",
                "random_number": random_num,
                "range": f"{min_val}-{max_val}",
                "timestamp": time.time()
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"生成随机数时出错: {str(e)}")
            return json.dumps({"operation": "generate_random_number", "error": str(e)})

    def generate_random_list(self, count: int = 5, min_val: int = 1, max_val: int = 100) -> str:
        """生成随机数列表

        Args:
            count (int): 要生成的随机数数量，默认为5
            min_val (int): 最小值，默认为1
            max_val (int): 最大值，默认为100

        Returns:
            str: 包含随机数列表和统计信息的JSON字符串

        Example:
            from agentica.tools.random_tool import RandomTool
            tool = RandomTool()
            result = tool.generate_random_list(3, 1, 10)
            print(result)
        """
        # 检查数量参数
        if count <= 0:
            logger.error(f"无效数量: {count}")
            return json.dumps({
                "operation": "generate_random_list", 
                "error": "数量必须大于0"
            })

        # 检查范围参数
        if min_val > max_val:
            logger.error(f"无效范围: min_val ({min_val}) > max_val ({max_val})")
            return json.dumps({
                "operation": "generate_random_list", 
                "error": "最小值不能大于最大值"
            })

        try:
            # 生成随机数列表
            random_list = [random.randint(min_val, max_val) for _ in range(count)]
            logger.info(f"生成 {count} 个随机数: {random_list}")
            
            # 计算统计信息并返回
            return json.dumps({
                "operation": "generate_random_list",
                "random_list": random_list,
                "count": count,
                "range": f"{min_val}-{max_val}",
                "average": round(sum(random_list) / len(random_list), 2),  # 平均值
                "max": max(random_list),        # 最大值
                "min": min(random_list),        # 最小值
                "sum": sum(random_list)         # 总和
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"生成随机数列表时出错: {str(e)}")
            return json.dumps({"operation": "generate_random_list", "error": str(e)})

    def generate_random_choice(self, choices: List[str]) -> str:
        """从选择列表中随机选择一个项目

        Args:
            choices (List[str]): 待选择的选项列表

        Returns:
            str: 包含选择结果的JSON字符串

        Example:
            from agentica.tools.random_tool import RandomTool
            tool = RandomTool()
            result = tool.generate_random_choice(["苹果", "香蕉", "橙子"])
            print(result)
        """
        # 检查选择列表是否为空
        if not choices:
            logger.error("选择列表为空")
            return json.dumps({
                "operation": "generate_random_choice", 
                "error": "选择列表不能为空"
            })

        try:
            # 随机选择一个项目
            selected_choice = random.choice(choices)
            logger.info(f"随机选择: {selected_choice} 来自 {choices}")
            
            # 返回选择结果和相关信息
            return json.dumps({
                "operation": "generate_random_choice",
                "selected_choice": selected_choice,     # 选中的项目
                "all_choices": choices,                 # 所有选项
                "total_choices": len(choices)           # 选项总数
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"随机选择时出错: {str(e)}")
            return json.dumps({"operation": "generate_random_choice", "error": str(e)})


if __name__ == '__main__':
    # 测试随机数工具的各项功能
    tool = RandomTool()
    print("测试 RandomTool:")
    print("1. 随机数:", tool.generate_random_number(1, 20))
    print("2. 随机列表:", tool.generate_random_list(3, 1, 10))
    print("3. 随机选择:", tool.generate_random_choice(["红色", "蓝色", "绿色", "黄色"])) 