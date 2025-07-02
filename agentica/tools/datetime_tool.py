# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: datetime tool for getting current date and time
"""
import json
from datetime import datetime, timezone
import pytz
from agentica.tools.base import Tool
from agentica.utils.log import logger


class DateTimeTool(Tool):
    def __init__(self):
        super().__init__(name="get_datetime_tool")
        self.register(self.get_current_datetime)
        self.register(self.get_current_date)
        self.register(self.get_current_time)
        self.register(self.get_datetime_in_timezone)

    def get_current_datetime(self, format_type: str = "full") -> str:
        """Get current date and time.

        Args:
            format_type (str): Format type - "full", "simple", "iso", "chinese"
                - full: 2025年7月2日 星期三 18:30:45 (北京时间)
                - simple: 2025-07-02 18:30:45
                - iso: 2025-07-02T18:30:45+08:00
                - chinese: 2025年7月2日 下午6点30分

        Returns:
            str: Current date and time in specified format
        """
        logger.info(f"get_current_datetime(format_type={format_type})")
        try:
            # 使用北京时间
            beijing_tz = pytz.timezone('Asia/Shanghai')
            now = datetime.now(beijing_tz)
            
            if format_type == "full":
                weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
                weekday = weekday_names[now.weekday()]
                result = f"{now.year}年{now.month}月{now.day}日 {weekday} {now.strftime('%H:%M:%S')} (北京时间)"
            elif format_type == "simple":
                result = now.strftime("%Y-%m-%d %H:%M:%S")
            elif format_type == "iso":
                result = now.isoformat()
            elif format_type == "chinese":
                hour = now.hour
                period = "上午" if hour < 12 else "下午"
                hour_12 = hour if hour <= 12 else hour - 12
                if hour_12 == 0:
                    hour_12 = 12
                result = f"{now.year}年{now.month}月{now.day}日 {period}{hour_12}点{now.minute}分"
            else:
                result = now.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.debug(f"Current datetime: {result}")
            return result
        except Exception as e:
            logger.error(f"Error getting current datetime: {str(e)}")
            return json.dumps({"operation": "get_current_datetime", "error": str(e)})

    def get_current_date(self, format_type: str = "chinese") -> str:
        """Get current date only.

        Args:
            format_type (str): Format type - "chinese", "iso", "us"
                - chinese: 2025年7月2日 星期三
                - iso: 2025-07-02
                - us: July 2, 2025

        Returns:
            str: Current date in specified format
        """
        logger.info(f"get_current_date(format_type={format_type})")
        try:
            beijing_tz = pytz.timezone('Asia/Shanghai')
            now = datetime.now(beijing_tz)
            
            if format_type == "chinese":
                weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
                weekday = weekday_names[now.weekday()]
                result = f"{now.year}年{now.month}月{now.day}日 {weekday}"
            elif format_type == "iso":
                result = now.strftime("%Y-%m-%d")
            elif format_type == "us":
                result = now.strftime("%B %d, %Y")
            else:
                result = f"{now.year}年{now.month}月{now.day}日"
            
            logger.debug(f"Current date: {result}")
            return result
        except Exception as e:
            logger.error(f"Error getting current date: {str(e)}")
            return json.dumps({"operation": "get_current_date", "error": str(e)})

    def get_current_time(self, format_type: str = "24h") -> str:
        """Get current time only.

        Args:
            format_type (str): Format type - "24h", "12h", "chinese"
                - 24h: 18:30:45
                - 12h: 6:30:45 PM
                - chinese: 下午6点30分45秒

        Returns:
            str: Current time in specified format
        """
        logger.info(f"get_current_time(format_type={format_type})")
        try:
            beijing_tz = pytz.timezone('Asia/Shanghai')
            now = datetime.now(beijing_tz)
            
            if format_type == "24h":
                result = now.strftime("%H:%M:%S")
            elif format_type == "12h":
                result = now.strftime("%I:%M:%S %p")
            elif format_type == "chinese":
                hour = now.hour
                period = "上午" if hour < 12 else "下午"
                hour_12 = hour if hour <= 12 else hour - 12
                if hour_12 == 0:
                    hour_12 = 12
                result = f"{period}{hour_12}点{now.minute}分{now.second}秒"
            else:
                result = now.strftime("%H:%M:%S")
            
            logger.debug(f"Current time: {result}")
            return result
        except Exception as e:
            logger.error(f"Error getting current time: {str(e)}")
            return json.dumps({"operation": "get_current_time", "error": str(e)})

    def get_datetime_in_timezone(self, timezone_name: str = "Asia/Shanghai") -> str:
        """Get current date and time in specified timezone.

        Args:
            timezone_name (str): Timezone name, eg: "Asia/Shanghai", "America/New_York", "Europe/London"

        Returns:
            str: Current datetime in specified timezone
        """
        logger.info(f"get_datetime_in_timezone(timezone_name={timezone_name})")
        try:
            tz = pytz.timezone(timezone_name)
            now = datetime.now(tz)
            
            result = f"{now.strftime('%Y年%m月%d日 %H:%M:%S')} ({timezone_name})"
            
            logger.debug(f"Datetime in {timezone_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error getting datetime in timezone {timezone_name}: {str(e)}")
            return json.dumps({"operation": "get_datetime_in_timezone", "error": str(e)})


if __name__ == '__main__':
    tool = DateTimeTool()
    print("=== 测试日期时间工具 ===")
    print(f"完整日期时间: {tool.get_current_datetime('full')}")
    print(f"简单格式: {tool.get_current_datetime('simple')}")
    print(f"中文日期: {tool.get_current_date('chinese')}")
    print(f"ISO日期: {tool.get_current_date('iso')}")
    print(f"24小时时间: {tool.get_current_time('24h')}")
    print(f"中文时间: {tool.get_current_time('chinese')}")
    print(f"北京时间: {tool.get_datetime_in_timezone('Asia/Shanghai')}")
    print(f"纽约时间: {tool.get_datetime_in_timezone('America/New_York')}") 