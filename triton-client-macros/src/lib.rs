use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput};

/// 自动为 protobuf 类型实现 PyO3 绑定
/// 
/// 这个宏只实现必要的类型转换 traits，不生成 getter/setter
/// 因为 protobuf 字段已经是 public 的，可以直接访问
/// 
/// 用法：
/// ```
/// #[derive(TritonPyClass)]
/// #[pyo3::pyclass]
/// pub struct MyMessage {
///     pub field: String,
/// }
/// ```
#[proc_macro_derive(TritonPyClass)]
pub fn triton_pyclass_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    
    let name = &input.ident;
    
    match &input.data {
        Data::Struct(_) => {
            generate_struct_impl(name)
        }
        Data::Enum(_) => {
            generate_enum_impl(name)
        }
        Data::Union(_) => {
            panic!("Union types are not supported")
        }
    }
}

fn generate_struct_impl(name: &syn::Ident) -> TokenStream {
    // 只生成类型转换 traits，不生成 getter/setter
    // 因为 protobuf 字段已经是 public 的
    let expanded = quote! {
        #[automatically_derived]
        const _: () = {
            use ::pyo3::prelude::*;
            use ::pyo3::types::PyAnyMethods;
            
            // 为 pyclass 提供默认构造函数
            #[::pyo3::pymethods]
            impl #name {
                #[new]
                fn __new__() -> Self {
                    Self::default()
                }
            }
            
            // 实现 IntoPyObject 以便返回给 Python
            impl<'py> ::pyo3::IntoPyObject<'py> for #name {
                type Target = ::pyo3::Bound<'py, Self>;
                type Output = ::pyo3::Bound<'py, Self>;
                type Error = ::std::convert::Infallible;

                fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::std::result::Result<Self::Output, Self::Error> {
                    Ok(::pyo3::Py::new(py, self).unwrap().into_bound(py))
                }
            }
            
            // 实现 FromPyObject 以便从 Python 接收
            impl<'py> ::pyo3::FromPyObject<'py> for #name {
                fn extract_bound(ob: &::pyo3::Bound<'py, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                    let obj = ob.downcast::<Self>()?;
                    Ok(obj.borrow().clone())
                }
            }
        };
    };

    TokenStream::from(expanded)
}

fn generate_enum_impl(name: &syn::Ident) -> TokenStream {
    let expanded = quote! {
        #[automatically_derived]
        const _: () = {
            use ::pyo3::prelude::*;
            use ::pyo3::types::PyAnyMethods;
            
            // 为 enum 实现类型转换
            impl<'py> ::pyo3::IntoPyObject<'py> for #name {
                type Target = ::pyo3::Bound<'py, Self>;
                type Output = ::pyo3::Bound<'py, Self>;
                type Error = ::std::convert::Infallible;

                fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::std::result::Result<Self::Output, Self::Error> {
                    Ok(::pyo3::Py::new(py, self).unwrap().into_bound(py))
                }
            }
            
            impl<'py> ::pyo3::FromPyObject<'py> for #name {
                fn extract_bound(ob: &::pyo3::Bound<'py, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                    let obj = ob.downcast::<Self>()?;
                    Ok(obj.borrow().clone())
                }
            }
        };
    };

    TokenStream::from(expanded)
}

