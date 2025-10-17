/// 示例：使用 TritonPyClass 宏
/// 
/// 这个文件展示了如何手动使用 TritonPyClass 宏为自定义类型添加 Python 绑定

use triton_client_macros::TritonPyClass;
use pyo3::prelude::*;

/// 示例 1: 简单结构体
#[derive(TritonPyClass, Clone, Default)]
pub struct SimpleMessage {
    pub name: String,
    pub value: i32,
}

/// 示例 2: 包含向量的结构体（自动 numpy 转换）
#[derive(TritonPyClass, Clone, Default)]
pub struct TensorData {
    pub name: String,
    pub shape: Vec<i64>,
    pub float_data: Vec<f32>,  // 自动转换为 numpy.ndarray[float32]
    pub int_data: Vec<i32>,    // 自动转换为 numpy.ndarray[int32]
}

/// 示例 3: 嵌套结构体
#[derive(TritonPyClass, Clone, Default)]
pub struct ModelRequest {
    pub model_name: String,
    pub inputs: Vec<TensorData>,
}

/// 示例 4: 手动添加额外的方法
#[pymethods]
impl ModelRequest {
    /// 自定义方法：验证请求
    fn validate(&self) -> PyResult<bool> {
        Ok(!self.model_name.is_empty() && !self.inputs.is_empty())
    }
    
    /// 自定义方法：添加输入
    fn add_input(&mut self, input: TensorData) {
        self.inputs.push(input);
    }
    
    /// 自定义方法：获取输入数量
    fn input_count(&self) -> usize {
        self.inputs.len()
    }
}

/// Python 使用示例（注释形式）
/// 
/// ```python
/// import triton_client
/// import numpy as np
/// 
/// # 创建简单消息
/// msg = SimpleMessage()
/// msg.name = "test"
/// msg.value = 42
/// 
/// # 创建张量数据（使用 numpy）
/// tensor = TensorData()
/// tensor.name = "input"
/// tensor.shape = [1, 224, 224, 3]
/// 
/// # 直接使用 numpy 数组！
/// tensor.float_data = np.random.rand(150528).astype(np.float32)
/// tensor.int_data = np.array([1, 2, 3, 4], dtype=np.int32)
/// 
/// # 创建模型请求
/// request = ModelRequest()
/// request.model_name = "resnet50"
/// request.add_input(tensor)
/// 
/// # 验证
/// if request.validate():
///     print(f"Request has {request.input_count()} inputs")
/// 
/// # 读取 numpy 数组
/// data = tensor.float_data  # 返回 numpy.ndarray
/// print(data.shape, data.dtype)
/// ```

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_creation() {
        let msg = SimpleMessage {
            name: "test".to_string(),
            value: 42,
        };
        assert_eq!(msg.name, "test");
        assert_eq!(msg.value, 42);
    }
    
    #[test]
    fn test_tensor_data() {
        let tensor = TensorData {
            name: "input".to_string(),
            shape: vec![1, 224, 224, 3],
            float_data: vec![1.0, 2.0, 3.0],
            int_data: vec![1, 2, 3],
        };
        assert_eq!(tensor.float_data.len(), 3);
        assert_eq!(tensor.int_data.len(), 3);
    }
}

