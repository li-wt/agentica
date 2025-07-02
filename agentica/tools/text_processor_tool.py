# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 文本处理工具，提供各种文本操作功能
"""
import json
import re
import hashlib
from typing import List, Dict, Any
from agentica.tools.base import Tool
from agentica.utils.log import logger


class TextProcessorTool(Tool):
    """文本处理工具
    
    提供多种文本处理功能，包括文本统计、信息提取、文本清理、
    格式转换、哈希生成和关键词提取等。支持中英文文本处理。
    """
    
    def __init__(self):
        """初始化文本处理工具"""
        super().__init__(name="text_processor_tool")
        # 注册所有文本处理功能函数
        self.register(self.count_words)         # 文本统计
        self.register(self.extract_emails)      # 邮箱提取
        self.register(self.extract_urls)        # URL提取
        self.register(self.clean_text)          # 文本清理
        self.register(self.transform_case)      # 大小写转换
        self.register(self.generate_hash)       # 哈希生成
        self.register(self.extract_keywords)    # 关键词提取

    def count_words(self, text: str) -> str:
        """统计文本中的字数、字符数和其他统计信息

        Args:
            text (str): 要分析的文本

        Returns:
            str: 包含文本统计信息的JSON字符串

        Example:
            from agentica.tools.text_processor_tool import TextProcessorTool
            tool = TextProcessorTool()
            result = tool.count_words("你好世界！这是一个测试。")
            print(result)
        """
        try:
            # 按空格分割单词（适用于英文，中文需要特殊处理）
            words = text.split()
            # 按句号分割句子
            sentences = text.split('.')
            # 按双换行分割段落
            paragraphs = text.split('\n\n')
            
            # 构建统计结果
            result = {
                "operation": "count_words",
                "original_text": text,
                "word_count": len(words),                                       # 单词数
                "character_count": len(text),                                   # 总字符数
                "character_count_no_spaces": len(text.replace(' ', '')),       # 不含空格字符数
                "sentence_count": len([s for s in sentences if s.strip()]),    # 句子数
                "paragraph_count": len([p for p in paragraphs if p.strip()]),  # 段落数
                # 平均单词长度（移除标点符号后计算）
                "average_word_length": round(sum(len(word.strip('.,!?;:"()[]')) for word in words) / len(words), 2) if words else 0
            }
            
            logger.info(f"文本分析: {len(words)} 个单词, {len(text)} 个字符")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"统计文本时出错: {str(e)}")
            return json.dumps({"operation": "count_words", "error": str(e)})

    def extract_emails(self, text: str) -> str:
        """从文本中提取邮箱地址

        Args:
            text (str): 要搜索邮箱地址的文本

        Returns:
            str: 包含找到的邮箱地址的JSON字符串

        Example:
            from agentica.tools.text_processor_tool import TextProcessorTool
            tool = TextProcessorTool()
            result = tool.extract_emails("联系我们：info@example.com 或 support@test.org")
            print(result)
        """
        try:
            # 邮箱地址正则表达式模式
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            
            result = {
                "operation": "extract_emails",
                "original_text": text,
                "emails": list(set(emails)),  # 去除重复项
                "count": len(set(emails))     # 唯一邮箱数量
            }
            
            logger.info(f"找到 {len(set(emails))} 个唯一邮箱地址")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"提取邮箱时出错: {str(e)}")
            return json.dumps({"operation": "extract_emails", "error": str(e)})

    def extract_urls(self, text: str) -> str:
        """Extract URLs from text.

        Args:
            text (str): The text to search for URLs.

        Returns:
            str: JSON string containing found URLs.

        Example:
            from agentica.tools.text_processor_tool import TextProcessorTool
            tool = TextProcessorTool()
            result = tool.extract_urls("Visit https://example.com or http://test.org")
            print(result)
        """
        try:
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, text)
            
            result = {
                "operation": "extract_urls",
                "original_text": text,
                "urls": list(set(urls)),  # Remove duplicates
                "count": len(set(urls))
            }
            
            logger.info(f"Found {len(set(urls))} unique URLs")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error extracting URLs: {str(e)}")
            return json.dumps({"operation": "extract_urls", "error": str(e)})

    def clean_text(self, text: str, remove_numbers: bool = False, remove_punctuation: bool = False, remove_extra_spaces: bool = True) -> str:
        """Clean text by removing unwanted characters.

        Args:
            text (str): The text to clean.
            remove_numbers (bool): Whether to remove numbers, default is False.
            remove_punctuation (bool): Whether to remove punctuation, default is False.
            remove_extra_spaces (bool): Whether to remove extra spaces, default is True.

        Returns:
            str: JSON string containing the cleaned text.

        Example:
            from agentica.tools.text_processor_tool import TextProcessorTool
            tool = TextProcessorTool()
            result = tool.clean_text("Hello!!!   World123??", remove_numbers=True, remove_punctuation=True)
            print(result)
        """
        try:
            cleaned = text
            
            if remove_numbers:
                cleaned = re.sub(r'\d+', '', cleaned)
            
            if remove_punctuation:
                cleaned = re.sub(r'[^\w\s]', '', cleaned)
            
            if remove_extra_spaces:
                cleaned = ' '.join(cleaned.split())
            
            result = {
                "operation": "clean_text",
                "original_text": text,
                "cleaned_text": cleaned,
                "settings": {
                    "remove_numbers": remove_numbers,
                    "remove_punctuation": remove_punctuation,
                    "remove_extra_spaces": remove_extra_spaces
                }
            }
            
            logger.info(f"Text cleaned: '{text}' -> '{cleaned}'")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return json.dumps({"operation": "clean_text", "error": str(e)})

    def transform_case(self, text: str, case_type: str = "upper") -> str:
        """Transform text case.

        Args:
            text (str): The text to transform.
            case_type (str): Type of transformation: 'upper', 'lower', 'title', 'capitalize', 'swapcase'.

        Returns:
            str: JSON string containing the transformed text.

        Example:
            from agentica.tools.text_processor_tool import TextProcessorTool
            tool = TextProcessorTool()
            result = tool.transform_case("hello world", "title")
            print(result)
        """
        try:
            case_map = {
                "upper": text.upper(),
                "lower": text.lower(),
                "title": text.title(),
                "capitalize": text.capitalize(),
                "swapcase": text.swapcase()
            }
            
            if case_type not in case_map:
                return json.dumps({
                    "operation": "transform_case",
                    "error": f"Invalid case_type: {case_type}. Valid options: {list(case_map.keys())}"
                })
            
            transformed = case_map[case_type]
            
            result = {
                "operation": "transform_case",
                "original_text": text,
                "transformed_text": transformed,
                "case_type": case_type
            }
            
            logger.info(f"Case transformed ({case_type}): '{text}' -> '{transformed}'")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error transforming case: {str(e)}")
            return json.dumps({"operation": "transform_case", "error": str(e)})

    def generate_hash(self, text: str, algorithm: str = "md5") -> str:
        """Generate hash of the text.

        Args:
            text (str): The text to hash.
            algorithm (str): Hash algorithm: 'md5', 'sha1', 'sha256', 'sha512'.

        Returns:
            str: JSON string containing the hash value.

        Example:
            from agentica.tools.text_processor_tool import TextProcessorTool
            tool = TextProcessorTool()
            result = tool.generate_hash("hello world", "sha256")
            print(result)
        """
        try:
            algorithm = algorithm.lower()
            
            if algorithm == "md5":
                hash_obj = hashlib.md5()
            elif algorithm == "sha1":
                hash_obj = hashlib.sha1()
            elif algorithm == "sha256":
                hash_obj = hashlib.sha256()
            elif algorithm == "sha512":
                hash_obj = hashlib.sha512()
            else:
                return json.dumps({
                    "operation": "generate_hash",
                    "error": f"Invalid algorithm: {algorithm}. Valid options: md5, sha1, sha256, sha512"
                })
            
            hash_obj.update(text.encode('utf-8'))
            hash_value = hash_obj.hexdigest()
            
            result = {
                "operation": "generate_hash",
                "original_text": text,
                "hash_value": hash_value,
                "algorithm": algorithm
            }
            
            logger.info(f"Generated {algorithm} hash for text")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error generating hash: {str(e)}")
            return json.dumps({"operation": "generate_hash", "error": str(e)})

    def extract_keywords(self, text: str, min_length: int = 3, max_keywords: int = 10) -> str:
        """Extract keywords from text (simple implementation).

        Args:
            text (str): The text to analyze.
            min_length (int): Minimum length of keywords, default is 3.
            max_keywords (int): Maximum number of keywords to return, default is 10.

        Returns:
            str: JSON string containing extracted keywords.

        Example:
            from agentica.tools.text_processor_tool import TextProcessorTool
            tool = TextProcessorTool()
            result = tool.extract_keywords("This is a simple text analysis example for keyword extraction")
            print(result)
        """
        try:
            # Simple stop words (can be expanded)
            stop_words = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
                "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
                "will", "would", "could", "should", "can", "may", "might", "must", "shall", "this", "that",
                "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"
            }
            
            # Clean and tokenize
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            
            # Filter words
            word_freq = {}
            for word in words:
                if len(word) >= min_length and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
            
            result = {
                "operation": "extract_keywords",
                "original_text": text,
                "keywords": [{"word": word, "frequency": freq} for word, freq in sorted_keywords],
                "total_unique_words": len(word_freq),
                "min_length": min_length,
                "max_keywords": max_keywords
            }
            
            logger.info(f"Extracted {len(sorted_keywords)} keywords")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return json.dumps({"operation": "extract_keywords", "error": str(e)})


if __name__ == '__main__':
    # 测试文本处理工具的各项功能
    tool = TextProcessorTool()
    print("测试 TextProcessorTool:")
    
    # 测试用的中英文混合文本
    test_text = "Hello World! 联系我们：info@example.com。访问 https://example.com 获取更多信息。这是一个包含123个数字的测试。"
    
    print("1. 字数统计:", tool.count_words(test_text))
    print("2. 提取邮箱:", tool.extract_emails(test_text))
    print("3. 提取URL:", tool.extract_urls(test_text))
    print("4. 清理文本:", tool.clean_text(test_text, remove_numbers=True, remove_punctuation=True))
    print("5. 大小写转换:", tool.transform_case("hello world", "title"))
    print("6. 生成哈希:", tool.generate_hash("hello world", "sha256"))
    print("7. 提取关键词:", tool.extract_keywords(test_text)) 