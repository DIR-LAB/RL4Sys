from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RL4SysAction(_message.Message):
    __slots__ = ("obs", "action", "mask", "reward", "data", "done", "reward_update_flag")
    class DataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    OBS_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    REWARD_UPDATE_FLAG_FIELD_NUMBER: _ClassVar[int]
    obs: bytes
    action: bytes
    mask: bytes
    reward: int
    data: _containers.ScalarMap[str, str]
    done: bool
    reward_update_flag: bool
    def __init__(self, obs: _Optional[bytes] = ..., action: _Optional[bytes] = ..., mask: _Optional[bytes] = ..., reward: _Optional[int] = ..., data: _Optional[_Mapping[str, str]] = ..., done: bool = ..., reward_update_flag: bool = ...) -> None: ...

class RL4SysActionList(_message.Message):
    __slots__ = ("actions",)
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    actions: _containers.RepeatedCompositeFieldContainer[RL4SysAction]
    def __init__(self, actions: _Optional[_Iterable[_Union[RL4SysAction, _Mapping]]] = ...) -> None: ...

class RL4SysModel(_message.Message):
    __slots__ = ("code", "model", "error")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    code: int
    model: bytes
    error: str
    def __init__(self, code: _Optional[int] = ..., model: _Optional[bytes] = ..., error: _Optional[str] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ActionResponse(_message.Message):
    __slots__ = ("code", "message")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    code: int
    message: str
    def __init__(self, code: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...
