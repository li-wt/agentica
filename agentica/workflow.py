# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from __future__ import annotations
import collections.abc
import inspect

from os import getenv
from uuid import uuid4
from types import GeneratorType
from typing import Any, Optional, Dict, Callable
from pydantic import BaseModel, Field, ConfigDict, field_validator, PrivateAttr

from agentica.utils.log import logger, set_log_level_to_debug
from agentica.agent import Agent
from agentica.run_response import RunResponse
from agentica.memory import WorkflowMemory, WorkflowRun
from agentica.storage.workflow.base import WorkflowStorage
from agentica.utils.misc import merge_dictionaries
from agentica.workflow_session import WorkflowSession


class Workflow(BaseModel):
    # -*- 工作流设置
    # 工作流名称
    name: Optional[str] = None
    # 工作流描述
    description: Optional[str] = None
    # 工作流 UUID（如果未设置则自动生成）
    workflow_id: Optional[str] = Field(None, validate_default=True)
    # 与此工作流关联的元数据
    workflow_data: Optional[Dict[str, Any]] = None

    # -*- 用户设置
    # 与此工作流交互的用户 ID
    user_id: Optional[str] = None
    # 与此工作流交互的用户相关元数据
    user_data: Optional[Dict[str, Any]] = None

    # -*- 会话设置
    # 会话 UUID（如果未设置则自动生成）
    session_id: Optional[str] = Field(None, validate_default=True)
    # 会话名称
    session_name: Optional[str] = None
    # 存储在数据库中的会话状态
    session_state: Dict[str, Any] = Field(default_factory=dict)

    # -*- 工作流记忆
    memory: WorkflowMemory = WorkflowMemory()

    # -*- 工作流存储
    storage: Optional[WorkflowStorage] = None
    # 来自数据库的 WorkflowSession：请勿手动设置
    _workflow_session: Optional[WorkflowSession] = None

    # debug_mode=True 启用调试日志
    debug_mode: bool = Field(False, validate_default=True)
    # monitoring=True 将工作流信息记录到 phidata.com
    monitoring: bool = getenv("PHI_MONITORING", "false").lower() == "true"
    # telemetry=True 记录最小遥测数据用于分析
    # 这有助于我们改进 Agent 并提供更好的支持
    telemetry: bool = getenv("PHI_TELEMETRY", "true").lower() == "true"

    # 请勿手动设置以下字段
    # 运行 ID：请勿手动设置
    run_id: Optional[str] = None
    # 工作流运行的输入：请勿手动设置
    run_input: Optional[Dict[str, Any]] = None
    # 工作流运行的响应：请勿手动设置
    run_response: RunResponse = Field(default_factory=RunResponse)
    # 与此会话关联的元数据：请勿手动设置
    session_data: Optional[Dict[str, Any]] = None

    # 由子类提供的 run 函数
    _subclass_run: Callable = PrivateAttr()
    # run 函数的参数
    _run_parameters: Dict[str, Any] = PrivateAttr()
    # run 函数的返回类型
    _run_return_type: Optional[str] = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @field_validator("workflow_id", mode="before")
    def set_workflow_id(cls, v: Optional[str]) -> str:
        workflow_id = v or str(uuid4())
        logger.debug(f"*********** 工作流 ID: {workflow_id} ***********")
        return workflow_id

    @field_validator("session_id", mode="before")
    def set_session_id(cls, v: Optional[str]) -> str:
        session_id = v or str(uuid4())
        logger.debug(f"*********** 工作流会话 ID: {session_id} ***********")
        return session_id

    @field_validator("debug_mode", mode="before")
    def set_log_level(cls, v: bool) -> bool:
        if v or getenv("PHI_DEBUG", "false").lower() == "true":
            set_log_level_to_debug()
            logger.debug("调试日志已启用")
        return v

    def get_workflow_data(self) -> Dict[str, Any]:
        workflow_data = self.workflow_data or {}
        if self.name is not None:
            workflow_data["name"] = self.name
        return workflow_data

    def get_session_data(self) -> Dict[str, Any]:
        session_data = self.session_data or {}
        if self.session_name is not None:
            session_data["session_name"] = self.session_name
        if len(self.session_state) > 0:
            session_data["session_state"] = self.session_state
        return session_data

    def get_workflow_session(self) -> WorkflowSession:
        """获取一个 WorkflowSession 对象，可以保存到数据库中"""

        return WorkflowSession(
            session_id=self.session_id,
            workflow_id=self.workflow_id,
            user_id=self.user_id,
            memory=self.memory.to_dict(),
            workflow_data=self.get_workflow_data(),
            user_data=self.user_data,
            session_data=self.get_session_data(),
        )

    def from_workflow_session(self, session: WorkflowSession):
        """从 WorkflowSession（来自数据库）加载现有的工作流"""

        # 从数据库获取 session_id、workflow_id 和 user_id
        if self.session_id is None and session.session_id is not None:
            self.session_id = session.session_id
        if self.workflow_id is None and session.workflow_id is not None:
            self.workflow_id = session.workflow_id
        if self.user_id is None and session.user_id is not None:
            self.user_id = session.user_id

        # 从数据库读取 workflow_data
        if session.workflow_data is not None:
            # 从数据库获取名称，如果未设置则更新工作流名称
            if self.name is None and "name" in session.workflow_data:
                self.name = session.workflow_data.get("name")

            # 如果工作流中设置了 workflow_data，则用工作流的 workflow_data 更新数据库的 workflow_data
            if self.workflow_data is not None:
                # 就地更新 workflow_session.workflow_data
                merge_dictionaries(session.workflow_data, self.workflow_data)
            self.workflow_data = session.workflow_data

        # 从数据库读取 user_data
        if session.user_data is not None:
            # 如果工作流中设置了 user_data，则用工作流的 user_data 更新数据库的 user_data
            if self.user_data is not None:
                # 就地更新 workflow_session.user_data
                merge_dictionaries(session.user_data, self.user_data)
            self.user_data = session.user_data

        # 从数据库读取 session_data
        if session.session_data is not None:
            # 从数据库获取 session_name，如果未设置则更新当前 session_name
            if self.session_name is None and "session_name" in session.session_data:
                self.session_name = session.session_data.get("session_name")

            # 从数据库获取 session_state 并更新当前 session_state
            if "session_state" in session.session_data:
                session_state_from_db = session.session_data.get("session_state")
                if (
                        session_state_from_db is not None
                        and isinstance(session_state_from_db, dict)
                        and len(session_state_from_db) > 0
                ):
                    # 如果已经设置了 session_state，则将数据库中的 session_state 与当前 session_state 合并
                    if len(self.session_state) > 0:
                        # 这会更新 session_state_from_db
                        merge_dictionaries(session_state_from_db, self.session_state)
                    # 更新当前 session_state
                    self.session_state = session_state_from_db

            # 如果工作流中设置了 session_data，则用工作流的 session_data 更新数据库的 session_data
            if self.session_data is not None:
                # 就地更新 workflow_session.session_data
                merge_dictionaries(session.session_data, self.session_data)
            self.session_data = session.session_data

        # 从数据库读取记忆
        if session.memory is not None:
            try:
                if "runs" in session.memory:
                    self.memory.runs = [WorkflowRun(**m) for m in session.memory["runs"]]
            except Exception as e:
                logger.warning(f"加载 WorkflowMemory 失败: {e}")
        logger.debug(f"-*- WorkflowSession 已加载: {session.session_id}")

    def read_from_storage(self) -> Optional[WorkflowSession]:
        """从存储中加载 WorkflowSession。

        Returns:
            Optional[WorkflowSession]: 加载的 WorkflowSession 或 None（如果未找到）。
        """
        if self.storage is not None and self.session_id is not None:
            self._workflow_session = self.storage.read(session_id=self.session_id)
            if self._workflow_session is not None:
                self.from_workflow_session(session=self._workflow_session)
        return self._workflow_session

    def write_to_storage(self) -> Optional[WorkflowSession]:
        """将 WorkflowSession 保存到存储

        Returns:
            Optional[WorkflowSession]: 保存的 WorkflowSession 或 None（如果未保存）。
        """
        if self.storage is not None:
            self._workflow_session = self.storage.upsert(session=self.get_workflow_session())
        return self._workflow_session

    def load_session(self, force: bool = False) -> Optional[str]:
        """从数据库加载现有会话并返回 session_id。
        如果会话不存在，则创建新会话。

        - 如果数据库中存在会话，则加载该会话。
        - 如果数据库中不存在会话，则创建新会话。
        """
        # 如果已经加载了 workflow_session，则从 workflow_session 返回 session_id
        # 如果 session_id 与 workflow_session 中的 session_id 匹配
        if self._workflow_session is not None and not force:
            if self.session_id is not None and self._workflow_session.session_id == self.session_id:
                return self._workflow_session.session_id

        # 加载现有会话或创建新会话
        if self.storage is not None:
            # 如果提供了 session_id，则加载现有会话
            logger.debug(f"读取 WorkflowSession: {self.session_id}")
            self.read_from_storage()

            # 如果会话不存在则创建新会话
            if self._workflow_session is None:
                logger.debug("-*- 创建新的 WorkflowSession")
                # write_to_storage() 将创建一个新的 WorkflowSession
                # 并用新会话填充 self._workflow_session
                self.write_to_storage()
                if self._workflow_session is None:
                    raise Exception("在存储中创建新的 WorkflowSession 失败")
                logger.debug(f"-*- 已创建 WorkflowSession: {self._workflow_session.session_id}")
                self.log_workflow_session()
        return self.session_id

    def run(self, *args: Any, **kwargs: Any):
        logger.error(f"{self.__class__.__name__}.run() 方法未实现。")
        return

    def run_workflow(self, *args: Any, **kwargs: Any):
        self.run_id = str(uuid4())
        self.run_input = {"args": args, "kwargs": kwargs}
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, workflow_id=self.workflow_id)
        self.read_from_storage()
        result = self._subclass_run(*args, **kwargs)

        # 情况 1：run 方法返回 Iterator[RunResponse]
        if isinstance(result, (GeneratorType, collections.abc.Iterator)):
            # 初始化 run_response 内容
            self.run_response.content = ""

            def result_generator():
                for item in result:
                    if isinstance(item, RunResponse):
                        # 更新 RunResponse 的 run_id、session_id 和 workflow_id
                        item.run_id = self.run_id
                        item.session_id = self.session_id
                        item.workflow_id = self.workflow_id

                        # 用结果中的内容更新 run_response
                        if item.content is not None and isinstance(item.content, str):
                            self.run_response.content += item.content
                    else:
                        logger.warning(f"Workflow.run() 应该只产生 RunResponse 对象，得到: {type(item)}")
                    yield item

                # 将运行添加到记忆中
                self.memory.add_run(WorkflowRun(input=self.run_input, response=self.run_response))
                # 将此运行写入数据库
                self.write_to_storage()
            return result_generator()
        # 情况 2：run 方法返回 RunResponse
        elif isinstance(result, RunResponse):
            # 用工作流运行的 run_id、session_id 和 workflow_id 更新结果
            result.run_id = self.run_id
            result.session_id = self.session_id
            result.workflow_id = self.workflow_id

            # 用结果中的内容更新 run_response
            if result.content is not None and isinstance(result.content, str):
                self.run_response.content = result.content

            # 将运行添加到记忆中
            self.memory.add_run(WorkflowRun(input=self.run_input, response=self.run_response))
            # 将此运行写入数据库
            self.write_to_storage()
            return result
        else:
            logger.warning(f"Workflow.run() 应该只返回 RunResponse 对象，得到: {type(result)}")
            return None

    def __init__(self, **data):
        super().__init__(**data)
        self.name = self.name or self.__class__.__name__
        # 检查子类是否提供了 'run'
        if self.__class__.run is not Workflow.run:
            # 存储绑定到实例的原始 run 方法
            self._subclass_run = self.__class__.run.__get__(self)
            # 获取 run 方法的参数
            sig = inspect.signature(self.__class__.run)
            # 将参数转换为可序列化格式
            self._run_parameters = {
                name: {
                    "name": name,
                    "default": param.default if param.default is not inspect.Parameter.empty else None,
                    "annotation": (
                        param.annotation.__name__
                        if hasattr(param.annotation, "__name__")
                        else (
                            str(param.annotation).replace("typing.Optional[", "").replace("]", "")
                            if "typing.Optional" in str(param.annotation)
                            else str(param.annotation)
                        )
                    )
                    if param.annotation is not inspect.Parameter.empty
                    else None,
                    "required": param.default is inspect.Parameter.empty,
                }
                for name, param in sig.parameters.items()
                if name != "self"
            }
            # 确定 run 方法的返回类型
            return_annotation = sig.return_annotation
            self._run_return_type = (
                return_annotation.__name__
                if return_annotation is not inspect.Signature.empty and hasattr(return_annotation, "__name__")
                else str(return_annotation)
                if return_annotation is not inspect.Signature.empty
                else None
            )
            # 用 run_workflow 替换实例的 run 方法
            object.__setattr__(self, "run", self.run_workflow.__get__(self))
        else:
            # 调用时这将记录错误
            self._subclass_run = self.run
            self._run_parameters = {}
            self._run_return_type = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        for field_name, field in self.__fields__.items():
            value = getattr(self, field_name)
            if isinstance(value, Agent):
                value.session_id = self.session_id

    def log_workflow_session(self):
        logger.debug(f"*********** 记录 WorkflowSession: {self.session_id} ***********")

    def rename_session(self, session_id: str, name: str):
        if self.storage is None:
            raise ValueError("存储未设置")
        workflow_session = self.storage.read(session_id)
        if workflow_session is None:
            raise Exception(f"WorkflowSession 未找到: {session_id}")
        if workflow_session.session_data is not None:
            workflow_session.session_data["session_name"] = name
        else:
            workflow_session.session_data = {"session_name": name}
        self.storage.upsert(workflow_session)

    def delete_session(self, session_id: str):
        if self.storage is None:
            raise ValueError("存储未设置")
        self.storage.delete_session(session_id)

    def deep_copy(self, *, update: Optional[Dict[str, Any]] = None) -> "Workflow":
        """创建并返回此工作流的深度副本，可选择更新字段。

        Args:
            update (Optional[Dict[str, Any]]): 新工作流的可选字段字典。

        Returns:
            Workflow: 新的工作流实例。
        """
        # 提取要为新工作流设置的字段
        fields_for_new_workflow = {}

        for field_name in self.model_fields_set:
            field_value = getattr(self, field_name)
            if field_value is not None:
                if isinstance(field_value, Agent):
                    fields_for_new_workflow[field_name] = field_value.deep_copy()
                else:
                    fields_for_new_workflow[field_name] = self._deep_copy_field(field_name, field_value)

        # 如果提供了更新，则更新字段
        if update:
            fields_for_new_workflow.update(update)

        # 创建新的工作流
        new_workflow = self.__class__(**fields_for_new_workflow)
        logger.debug(
            f"已创建新的工作流: workflow_id: {new_workflow.workflow_id} | session_id: {new_workflow.session_id}"
        )
        return new_workflow

    def _deep_copy_field(self, field_name: str, field_value: Any) -> Any:
        """根据字段类型深度复制字段的辅助方法。"""
        from copy import copy, deepcopy

        # 对于记忆，使用其 deep_copy 方法
        if field_name == "memory":
            return field_value.deep_copy()

        # 对于复合类型，尝试深度复制
        if isinstance(field_value, (list, dict, set, WorkflowStorage)):
            try:
                return deepcopy(field_value)
            except Exception as e:
                logger.warning(f"深度复制字段失败: {field_name} - {e}")
                try:
                    return copy(field_value)
                except Exception as e:
                    logger.warning(f"复制字段失败: {field_name} - {e}")
                    return field_value

        # 对于 pydantic 模型，尝试深度复制
        if isinstance(field_value, BaseModel):
            try:
                return field_value.model_copy(deep=True)
            except Exception as e:
                logger.warning(f"深度复制字段失败: {field_name} - {e}")
                try:
                    return field_value.model_copy(deep=False)
                except Exception as e:
                    logger.warning(f"复制字段失败: {field_name} - {e}")
                    return field_value

        # 对于其他类型，原样返回
        return field_value
