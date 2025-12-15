# TritonClient Type Stubs

这个目录包含了 `tritonclient` 库的类型存根文件（`.pyi`），用于提供类型提示和 IDE 支持。

## 文件结构

```
tritonclient/
├── __init__.pyi
├── grpc/
│   └── __init__.pyi          # tritonclient.grpc 模块的类型存根
├── utils/
│   ├── __init__.pyi           # tritonclient.utils 模块的类型存根
│   └── shared_memory.pyi      # tritonclient.utils.shared_memory 模块的类型存根
└── README.md
```

## 包含的接口和类型

### tritonclient.grpc

- **InferenceServerClient**: 主要的推理服务器客户端类
  - `__init__()`: 初始化客户端
  - `get_model_metadata()`: 获取模型元数据
  - `get_model_config()`: 获取模型配置
  - `get_server_metadata()`: 获取服务器元数据
  - `register_system_shared_memory()`: 注册系统共享内存
  - `unregister_system_shared_memory()`: 注销系统共享内存
  - `infer()`: 同步推理
  - `async_infer()`: 异步推理
  - `close()`: 关闭客户端连接

- **InferInput**: 推理输入张量类
  - `__init__()`: 初始化输入
  - `set_shape()`: 设置形状
  - `set_data_from_numpy()`: 从 numpy 数组设置数据
  - `set_shared_memory()`: 设置共享内存

- **InferRequestedOutput**: 推理输出请求类
  - `__init__()`: 初始化输出请求
  - `set_shared_memory()`: 设置共享内存

- **InferResult**: 推理结果类
  - `as_numpy()`: 获取 numpy 数组
  - `get_output()`: 获取输出对象

- **ModelMetadata**: 模型元数据类
- **ModelConfig**: 模型配置类
- **CallContext**: 异步调用上下文类

### tritonclient.utils

- **triton_to_np_dtype()**: 将 Triton 数据类型转换为 numpy 数据类型

### tritonclient.utils.shared_memory

- **create_shared_memory_region()**: 创建共享内存区域
- **set_shared_memory_region()**: 设置共享内存区域内容
- **get_contents_as_numpy()**: 获取共享内存内容为 numpy 数组
- **destroy_shared_memory_region()**: 销毁共享内存区域
- **mapped_shared_memory_regions()**: 获取所有映射的共享内存区域

## 使用说明

这些类型存根文件基于 `src/model/triton` 和 `src/triton` 目录中实际使用的接口生成。它们提供了完整的类型信息，有助于：

1. IDE 自动补全和类型检查
2. 静态类型检查工具（如 mypy）的类型验证
3. 代码文档和可读性

## 注意事项

- 这些类型存根仅包含项目中实际使用的接口和类型
- 如果添加了新的 tritonclient 接口使用，需要相应更新这些类型存根文件
- 类型存根可能与实际的 tritonclient 库实现略有差异，但应该覆盖所有使用的接口

