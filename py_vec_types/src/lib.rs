//! Python-bound Vec type wrappers provide List<T> types for easier manipulation of Vec fields in Python.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::ops::{Deref, DerefMut};

/// A mutable list type used to set Vec fields.
///
/// It can be constructed from Python list, NumPy array, etc., and supports CRUD operations.
///
/// Note: Since pyo3 does not support generic pyclass, we have created concrete types for each type.
///
macro_rules! define_list_type {
    ($name:ident, $t:ty, $py_array:ty, $py_readonly:ty) => {
        #[pyclass(module="triton_client")]
        #[derive(Debug, Clone)]
        pub struct $name {
            inner: Vec<$t>,
        }

        impl $name {
            pub fn new(inner: Vec<$t>) -> Self {
                Self { inner }
            }

            pub fn into_vec(self) -> Vec<$t> {
                self.inner
            }
        }

        impl Deref for $name {
            type Target = Vec<$t>;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.inner
            }
        }

        impl AsRef<Vec<$t>> for $name {
            fn as_ref(&self) -> &Vec<$t> {
                &self.inner
            }
        }

        #[pymethods]
        impl $name {
            /// from Python list construct
            #[new]
            fn new_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
                if let Ok(list) = obj.cast::<PyList>() {
                    let vec: Vec<$t> = list.extract()?;
                    Ok(Self { inner: vec })
                } else {
                    // 尝试直接提取
                    let vec: Vec<$t> = obj.extract()?;
                    Ok(Self { inner: vec })
                }
            }

            /// from numpy array construct
            #[staticmethod]
            fn from_array(arr: $py_readonly) -> PyResult<Self> {
                Ok(Self {
                    inner: arr.to_vec()?,
                })
            }

            /// push item
            fn append(&mut self, item: $t) {
                self.inner.push(item);
            }

            /// len
            fn __len__(&self) -> usize {
                self.inner.len()
            }

            fn __getitem__(&self, index: usize) -> PyResult<$t> {
                self.inner
                    .get(index)
                    .copied()
                    .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Index out of range"))
            }

            fn __setitem__(&mut self, index: usize, value: $t) -> PyResult<()> {
                if let Some(elem) = self.inner.get_mut(index) {
                    *elem = value;
                    Ok(())
                } else {
                    Err(pyo3::exceptions::PyIndexError::new_err(
                        "Index out of range",
                    ))
                }
            }

            /// remove item
            fn remove(&mut self, index: usize) -> PyResult<$t> {
                if index < self.inner.len() {
                    Ok(self.inner.remove(index))
                } else {
                    Err(pyo3::exceptions::PyIndexError::new_err(
                        "Index out of range",
                    ))
                }
            }

            /// insert item
            fn insert(&mut self, index: usize, value: $t) -> PyResult<()> {
                if index <= self.inner.len() {
                    self.inner.insert(index, value);
                    Ok(())
                } else {
                    Err(pyo3::exceptions::PyIndexError::new_err(
                        "Index out of range",
                    ))
                }
            }

            /// to Python list
            fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
                PyList::new(py, &self.inner)
            }

            /// to numpy array
            fn to_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, $py_array>> {
                Ok(<$py_array>::from_vec(py, self.inner.clone()))
            }

            /// move to numpy array
            fn into_array<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, $py_array>> {
                Ok(<$py_array>::from_vec(py, std::mem::take(&mut self.inner)))
            }

            /// clear items
            fn clear(&mut self) {
                self.inner.clear();
            }

            /// deep copy
            fn copy(&self) ->Self {
                self.clone()
            }
        }
    };
}

// 为支持的数值类型定义 List 类型
define_list_type!(ListBool, bool, PyArray1<bool>, PyReadonlyArray1<bool>);
define_list_type!(ListI8, i8, PyArray1<i8>, PyReadonlyArray1<i8>);
define_list_type!(ListI16, i16, PyArray1<i16>, PyReadonlyArray1<i16>);
define_list_type!(ListI32, i32, PyArray1<i32>, PyReadonlyArray1<i32>);
define_list_type!(ListI64, i64, PyArray1<i64>, PyReadonlyArray1<i64>);
define_list_type!(ListU8, u8, PyArray1<u8>, PyReadonlyArray1<u8>);
define_list_type!(ListU16, u16, PyArray1<u16>, PyReadonlyArray1<u16>);
define_list_type!(ListU32, u32, PyArray1<u32>, PyReadonlyArray1<u32>);
define_list_type!(ListU64, u64, PyArray1<u64>, PyReadonlyArray1<u64>);
define_list_type!(ListF32, f32, PyArray1<f32>, PyReadonlyArray1<f32>);
define_list_type!(ListF64, f64, PyArray1<f64>, PyReadonlyArray1<f64>);
