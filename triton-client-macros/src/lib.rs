use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type, PathArguments, GenericArgument};

/// 自动为 protobuf 类型实现 PyO3 绑定
/// 
/// 用法：
/// ```
/// #[derive(TritonPyClass)]
/// pub struct MyMessage {
///     pub field: String,
///     pub numbers: Vec<f32>,  // 自动转换为 numpy
/// }
/// ```
#[proc_macro_derive(TritonPyClass, attributes(pyo3))]
pub fn triton_pyclass_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    
    let name = &input.ident;
    let generics = &input.generics;
    
    match &input.data {
        Data::Struct(data_struct) => {
            generate_struct_impl(name, generics, data_struct)
        }
        Data::Enum(_) => {
            // 对于 enum，生成简单的 PyClass 实现
            generate_enum_impl(name, generics)
        }
        Data::Union(_) => {
            panic!("Union types are not supported")
        }
    }
}

fn generate_struct_impl(
    name: &syn::Ident,
    _generics: &syn::Generics,
    data_struct: &syn::DataStruct,
) -> TokenStream {
    let fields = match &data_struct.fields {
        Fields::Named(fields) => &fields.named,
        _ => panic!("Only named fields are supported"),
    };

    // 分析字段，区分普通字段和需要特殊处理的字段
    let mut field_getters = Vec::new();
    let mut field_setters = Vec::new();
    let mut numpy_conversions = Vec::new();
    
    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_type = &field.ty;
        
        // 检查是否是 Vec<primitive>
        if let Some(inner_type) = extract_vec_inner_type(field_type) {
            if is_primitive_type(&inner_type) {
                // 生成 numpy 转换的 getter/setter
                let numpy_getter = generate_numpy_getter(field_name, &inner_type);
                let numpy_setter = generate_numpy_setter(field_name, &inner_type);
                
                field_getters.push(numpy_getter);
                field_setters.push(numpy_setter);
                
                numpy_conversions.push(field_name);
                continue;
            }
        }
        
        // 普通字段的 getter/setter
        field_getters.push(quote! {
            #[getter]
            fn #field_name(&self) -> ::pyo3::PyResult<#field_type> {
                Ok(self.#field_name.clone())
            }
        });
        
        field_setters.push(quote! {
            #[setter]
            fn #field_name(&mut self, value: #field_type) {
                self.#field_name = value;
            }
        });
    }

    let expanded = quote! {
        #[automatically_derived]
        const _: () = {
            use ::pyo3::prelude::*;
            use ::pyo3::types::PyAnyMethods;
            
            #[::pyo3::pymethods]
            impl #name {
                #[new]
                fn __new__() -> Self {
                    Self::default()
                }
                
                #(#field_getters)*
                #(#field_setters)*
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

fn generate_enum_impl(
    name: &syn::Ident,
    _generics: &syn::Generics,
) -> TokenStream {
    let expanded = quote! {
        #[automatically_derived]
        #[::pyo3::pyclass]
        impl #name {}
        
        impl<'py> ::pyo3::IntoPyObject<'py> for #name {
            type Target = ::pyo3::Bound<'py, Self>;
            type Output = ::pyo3::Bound<'py, Self>;
            type Error = ::std::convert::Infallible;

            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::std::result::Result<Self::Output, Self::Error> {
                Ok(::pyo3::Py::new(py, self).unwrap().into_bound(py))
            }
        }
    };

    TokenStream::from(expanded)
}

fn generate_numpy_getter(field_name: &syn::Ident, inner_type: &str) -> proc_macro2::TokenStream {
    let inner_type_ident: proc_macro2::TokenStream = inner_type.parse().unwrap();
    
    quote! {
        #[getter]
        fn #field_name<'py>(&self, py: ::pyo3::Python<'py>) -> ::pyo3::PyResult<::numpy::PyArray1<#inner_type_ident>> {
            Ok(::numpy::PyArray1::from_slice(py, &self.#field_name))
        }
    }
}

fn generate_numpy_setter(field_name: &syn::Ident, inner_type: &str) -> proc_macro2::TokenStream {
    let inner_type_ident: proc_macro2::TokenStream = inner_type.parse().unwrap();
    
    quote! {
        #[setter]
        fn #field_name(&mut self, value: ::numpy::PyReadonlyArray1<#inner_type_ident>) {
            self.#field_name = value.as_slice().unwrap().to_vec();
        }
    }
}

fn extract_vec_inner_type(ty: &Type) -> Option<String> {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            if segment.ident == "Vec" {
                if let PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(GenericArgument::Type(Type::Path(inner_path))) = args.args.first() {
                        if let Some(inner_segment) = inner_path.path.segments.last() {
                            return Some(inner_segment.ident.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

fn is_primitive_type(type_name: &str) -> bool {
    matches!(
        type_name,
        "f32" | "f64" | "i32" | "i64" | "u32" | "u64" | "bool" | "i8" | "u8" | "i16" | "u16"
    )
}

