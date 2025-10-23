use crate::inference::InferTensorContents;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

#[pymethods]
impl InferTensorContents {
    #[pyo3(signature = (src=None))]
    fn replace_bool_contents<'py>(
        &mut self,
        py: Python<'py>,
        src: Option<PyReadonlyArray1<bool>>,
    ) -> crate::Result<Bound<'py, PyArray1<bool>>> {
        let src = match src {
            None => {
                vec![]
            }
            Some(src) => src.to_vec()?,
        };
        let contents = std::mem::replace(&mut self.bool_contents, src);
        Ok(PyArray1::from_vec(py, contents))
    }
    #[pyo3(signature = (src=None))]
    fn replace_int_contents<'py>(
        &mut self,
        py: Python<'py>,
        src: Option<PyReadonlyArray1<i32>>,
    ) -> crate::Result<Bound<'py, PyArray1<i32>>> {
        let src = match src {
            None => {
                vec![]
            }
            Some(src) => src.to_vec()?,
        };
        let contents = std::mem::replace(&mut self.int_contents, src);
        Ok(PyArray1::from_vec(py, contents))
    }
    #[pyo3(signature = (src=None))]
    fn replace_int64_contents<'py>(
        &mut self,
        py: Python<'py>,
        src: Option<PyReadonlyArray1<i64>>,
    ) -> crate::Result<Bound<'py, PyArray1<i64>>> {
        let src = match src {
            None => {
                vec![]
            }
            Some(src) => src.to_vec()?,
        };
        let contents = std::mem::replace(&mut self.int64_contents, src);
        Ok(PyArray1::from_vec(py, contents))
    }
    #[pyo3(signature = (src=None))]
    fn replace_uint_contents<'py>(
        &mut self,
        py: Python<'py>,
        src: Option<PyReadonlyArray1<u32>>,
    ) -> crate::Result<Bound<'py, PyArray1<u32>>> {
        let src = match src {
            None => {
                vec![]
            }
            Some(src) => src.to_vec()?,
        };
        let contents = std::mem::replace(&mut self.uint_contents, src);
        Ok(PyArray1::from_vec(py, contents))
    }
    #[pyo3(signature = (src=None))]
    fn replace_uint64_contents<'py>(
        &mut self,
        py: Python<'py>,
        src: Option<PyReadonlyArray1<u64>>,
    ) -> crate::Result<Bound<'py, PyArray1<u64>>> {
        let src = match src {
            None => {
                vec![]
            }
            Some(src) => src.to_vec()?,
        };
        let contents = std::mem::replace(&mut self.uint64_contents, src);
        Ok(PyArray1::from_vec(py, contents))
    }
    #[pyo3(signature = (src=None))]
    fn replace_fp32_contents<'py>(
        &mut self,
        py: Python<'py>,
        src: Option<PyReadonlyArray1<f32>>,
    ) -> crate::Result<Bound<'py, PyArray1<f32>>> {
        let src = match src {
            None => {
                vec![]
            }
            Some(src) => src.to_vec()?,
        };
        let contents = std::mem::replace(&mut self.fp32_contents, src);
        Ok(PyArray1::from_vec(py, contents))
    }
    #[pyo3(signature = (src=None))]
    fn replace_fp64_contents<'py>(
        &mut self,
        py: Python<'py>,
        src: Option<PyReadonlyArray1<f64>>,
    ) -> crate::Result<Bound<'py, PyArray1<f64>>> {
        let src = match src {
            None => {
                vec![]
            }
            Some(src) => src.to_vec()?,
        };
        let contents = std::mem::replace(&mut self.fp64_contents, src);
        Ok(PyArray1::from_vec(py, contents))
    }
    #[pyo3(signature = (src=None))]
    fn replace_bytes_contents<'py>(
        &mut self,
        py: Python<'py>,
        src: Option<PyReadonlyArray2<u8>>,
    ) -> crate::Result<Bound<'py, PyArray2<u8>>> {
        let src = match src {
            None => {
                vec![]
            }
            Some(src) => {
                let shape = src.shape();
                let num_cols = shape[1];
                let a = src.as_slice()?;
                a.chunks_exact(num_cols)
                    .map(|chunk| chunk.to_vec())
                    .collect()
            }
        };
        let contents = std::mem::replace(&mut self.bytes_contents, src);
        Ok(PyArray2::from_vec2(py, contents.as_slice())?)
    }
}
