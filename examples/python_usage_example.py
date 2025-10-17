"""
Triton Client Python Usage Examples

这个文件展示了如何使用 triton_client Python 绑定与 NVIDIA Triton Inference Server 交互
"""

import triton_client
import json
import numpy as np

def example_server_health_check():
    """示例：检查服务器健康状态"""
    print("=== Server Health Check ===")
    
    # 创建客户端连接
    client = triton_client.Client("http://localhost:8000")
    
    # 检查服务器是否存活
    is_live = client.server_live()
    print(f"Server Live: {is_live}")
    
    # 检查服务器是否就绪
    is_ready = client.server_ready()
    print(f"Server Ready: {is_ready}")
    
    # 获取服务器元数据
    metadata_json = client.server_metadata()
    metadata = json.loads(metadata_json)
    print(f"Server Name: {metadata['name']}")
    print(f"Server Version: {metadata['version']}")
    print(f"Extensions: {metadata['extensions']}")


def example_model_operations():
    """示例：模型操作"""
    print("\n=== Model Operations ===")
    
    client = triton_client.Client("http://localhost:8000")
    model_name = "my_model"
    model_version = "1"  # 可选，留空则使用最新版本
    
    # 检查模型是否就绪
    req = triton_client.ModelReadyRequest(model_name, model_version)
    is_ready = client.model_ready(req)
    print(f"Model '{model_name}' Ready: {is_ready}")
    
    # 获取模型元数据
    req = triton_client.ModelMetadataRequest(model_name, model_version)
    metadata_json = client.model_metadata(req)
    metadata = json.loads(metadata_json)
    print(f"Model Platform: {metadata['platform']}")
    print(f"Inputs: {metadata['inputs']}")
    print(f"Outputs: {metadata['outputs']}")
    
    # 获取模型配置
    req = triton_client.ModelConfigRequest(model_name, model_version)
    config_json = client.model_config(req)
    config = json.loads(config_json)
    print(f"Model Config: {config}")
    
    # 获取模型统计信息
    req = triton_client.ModelStatisticsRequest(model_name, model_version)
    stats_json = client.model_statistics(req)
    stats = json.loads(stats_json)
    print(f"Model Statistics: {stats}")


def example_model_inference():
    """示例：模型推理"""
    print("\n=== Model Inference ===")
    
    client = triton_client.Client("http://localhost:8000")
    model_name = "my_model"
    
    # 创建推理请求
    infer_req = triton_client.ModelInferRequest(
        model_name=model_name,
        model_version="",  # 使用最新版本
        id="inference_1"
    )
    
    # 定义输入张量
    # 假设模型接受 shape=(1, 224, 224, 3) 的 FP32 输入
    input_tensor = triton_client.InferInput(
        name="input",
        datatype="FP32",
        shape=[1, 224, 224, 3]
    )
    infer_req.add_input(input_tensor)
    
    # 指定需要的输出
    output_tensor = triton_client.InferRequestedOutputTensor(name="output")
    infer_req.add_output(output_tensor)
    
    # 准备输入数据
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    input_bytes = input_data.tobytes()
    infer_req.add_raw_input(input_bytes)
    
    # 执行推理
    try:
        response_bytes = client.model_infer(infer_req)
        print(f"Inference successful! Response size: {len(response_bytes)} bytes")
        # 可以使用 protobuf 解码 response_bytes 来获取结果
    except Exception as e:
        print(f"Inference failed: {e}")


def example_repository_management():
    """示例：模型仓库管理"""
    print("\n=== Repository Management ===")
    
    client = triton_client.Client("http://localhost:8000")
    
    # 获取仓库索引
    req = triton_client.RepositoryIndexRequest(
        repository_name="",  # 空字符串表示所有仓库
        ready=True  # 只显示就绪的模型
    )
    index_json = client.repository_index(req)
    index = json.loads(index_json)
    print(f"Repository Models: {index['models']}")
    
    # 加载模型
    model_name = "my_model"
    req = triton_client.RepositoryModelLoadRequest(
        model_name=model_name,
        repository_name=""
    )
    try:
        success = client.repository_model_load(req)
        print(f"Model '{model_name}' loaded: {success}")
    except Exception as e:
        print(f"Failed to load model: {e}")
    
    # 卸载模型
    req = triton_client.RepositoryModelUnloadRequest(
        model_name=model_name,
        repository_name=""
    )
    try:
        success = client.repository_model_unload(req)
        print(f"Model '{model_name}' unloaded: {success}")
    except Exception as e:
        print(f"Failed to unload model: {e}")


def example_shared_memory():
    """示例：共享内存管理"""
    print("\n=== Shared Memory Management ===")
    
    client = triton_client.Client("http://localhost:8000")
    
    # 系统共享内存状态
    req = triton_client.SystemSharedMemoryStatusRequest(name="")
    status_json = client.system_shared_memory_status(req)
    status = json.loads(status_json)
    print(f"System Shared Memory Regions: {status['regions']}")
    
    # 注册系统共享内存
    req = triton_client.SystemSharedMemoryRegisterRequest(
        name="my_shm",
        key="/my_shm_key",
        offset=0,
        byte_size=1024
    )
    try:
        success = client.system_shared_memory_register(req)
        print(f"System shared memory registered: {success}")
    except Exception as e:
        print(f"Failed to register system shared memory: {e}")
    
    # 取消注册系统共享内存
    req = triton_client.SystemSharedMemoryUnregisterRequest(name="my_shm")
    try:
        success = client.system_shared_memory_unregister(req)
        print(f"System shared memory unregistered: {success}")
    except Exception as e:
        print(f"Failed to unregister system shared memory: {e}")
    
    # CUDA 共享内存状态
    req = triton_client.CudaSharedMemoryStatusRequest(name="")
    status_json = client.cuda_shared_memory_status(req)
    status = json.loads(status_json)
    print(f"CUDA Shared Memory Regions: {status['regions']}")


def example_trace_settings():
    """示例：跟踪设置"""
    print("\n=== Trace Settings ===")
    
    client = triton_client.Client("http://localhost:8000")
    
    # 获取跟踪设置
    req = triton_client.TraceSettingRequest(model_name="")
    try:
        settings_json = client.trace_setting(req)
        settings = json.loads(settings_json)
        print(f"Trace Settings: {settings}")
    except Exception as e:
        print(f"Failed to get trace settings: {e}")


def example_with_authentication():
    """示例：使用认证token"""
    print("\n=== With Authentication ===")
    
    # 使用Bearer token进行认证
    token = "your_access_token_here"
    client = triton_client.Client("https://your-triton-server.com:8001", token=token)
    
    is_live = client.server_live()
    print(f"Authenticated Server Live: {is_live}")


if __name__ == "__main__":
    print("Triton Client Python Usage Examples")
    print("=" * 50)
    
    # 注意：这些示例假设 Triton Inference Server 正在 localhost:8000 运行
    # 请根据实际情况修改服务器地址和模型名称
    
    try:
        example_server_health_check()
        example_model_operations()
        # example_model_inference()  # 需要实际的模型和数据
        # example_repository_management()  # 需要实际的模型
        # example_shared_memory()  # 需要系统共享内存支持
        # example_trace_settings()
        # example_with_authentication()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure Triton Inference Server is running and accessible.")

