use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type, Variant};

/// 自动为 protobuf 类型生成 PyO3 构造函数
///
/// 这个宏为生成的 protobuf 类型添加：
/// - 对于 **struct**: 带所有字段可选参数的 `__new__` 构造函数
/// - 对于 **C-style enum**: 为每个变体生成 staticmethod
/// - 对于 **oneof enum**: 为每个变体生成接收参数的 classmethod
///
/// 注意：类型转换由 `#[pyo3::pyclass(get_all, set_all)]` 自动处理
///
/// # 示例
///
/// ## Struct:
/// ```python
/// config = ModelConfig(name="model1", max_batch_size=8)
/// ```
///
/// ## C-style Enum:
/// ```python
/// dtype = DataType.type_int32()  # 返回 DataType::TypeInt32
/// ```
///
/// ## Oneof Enum:
/// ```python
/// policy = PolicyChoice.latest(latest_obj)
/// ```
#[proc_macro_derive(ImplPyNew)]
pub fn triton_pyclass_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;

    match &input.data {
        Data::Struct(data_struct) => generate_struct_impl(name, &data_struct.fields),
        Data::Enum(data_enum) => generate_enum_impl(name, &data_enum.variants),
        Data::Union(_) => {
            panic!("Union types are not supported")
        }
    }
}

fn generate_struct_impl(name: &syn::Ident, fields: &Fields) -> TokenStream {
    // 提取所有字段信息
    let field_info: Vec<_> = match fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .map(|f| {
                let field_name = f.ident.as_ref().unwrap();
                let field_type = &f.ty;
                (field_name.clone(), field_type.clone())
            })
            .collect(),
        _ => vec![],
    };

    // 生成构造函数参数列表
    let param_list = field_info.iter().map(|(name, ty)| {
        quote! { #name: #ty }
    });

    // 生成参数名列表（用于 signature）
    let param_names = field_info.iter().map(|(name, _)| {
        quote! { #name }
    });

    // 生成字段初始化代码
    let field_init = field_info.iter().map(|(name, _ty)| {
        quote! { #name: #name }
    });

    let expanded = quote! {
        #[automatically_derived]
        const _: () = {
            use ::pyo3::prelude::*;
            use ::pyo3::types::PyAnyMethods;

            // 为 pyclass 提供带可选参数的构造函数
            #[::pyo3::pymethods]
            impl #name {
                #[new]
                #[pyo3(signature=(#(#param_names),*))]
                fn __new__(
                    #(#param_list),*
                ) -> Self {
                    Self {
                        #(#field_init),*
                    }
                }
            }
        };
    };

    TokenStream::from(expanded)
}


fn generate_enum_impl(
    name: &syn::Ident,
    variants: &syn::punctuated::Punctuated<Variant, syn::token::Comma>,
) -> TokenStream {
    // 检查是否是 C-style enum（所有变体都没有字段）还是 oneof enum（变体有字段）
    let has_fields = variants.iter().any(|v| !matches!(v.fields, Fields::Unit));

    if has_fields {
        // Oneof enum: 为每个变体生成接收参数的构造方法
        generate_oneof_enum_impl(name, variants)
    } else {
        // C-style enum: 为每个变体生成无参数的构造方法
        generate_simple_enum_impl(name, variants)
    }
}

fn generate_simple_enum_impl(
    name: &syn::Ident,
    variants: &syn::punctuated::Punctuated<Variant, syn::token::Comma>,
) -> TokenStream {
    // 为每个 C-style enum 变体生成一个 staticmethod
    let variant_constructors = variants.iter().map(|variant| {
        let variant_name = &variant.ident;
        // 将变体名转换为 snake_case 作为方法名
        let method_name = syn::Ident::new(
            &to_snake_case(&variant_name.to_string()),
            variant_name.span(),
        );

        quote! {
            #[staticmethod]
            fn #method_name() -> Self {
                Self::#variant_name
            }
        }
    });

    let expanded = quote! {
        #[automatically_derived]
        const _: () = {
            use ::pyo3::prelude::*;

            // 为每个 C-style enum 变体生成构造方法
            #[::pyo3::pymethods]
            impl #name {
                #(#variant_constructors)*
            }
        };
    };

    TokenStream::from(expanded)
}

fn generate_oneof_enum_impl(
    name: &syn::Ident,
    variants: &syn::punctuated::Punctuated<Variant, syn::token::Comma>,
) -> TokenStream {
    // 为每个 oneof 变体生成一个构造方法（classmethod）
    let variant_constructors = variants.iter().map(|variant| {
        let variant_name = &variant.ident;
        // 将变体名转换为 snake_case 作为方法名
        let method_name = syn::Ident::new(
            &to_snake_case(&variant_name.to_string()),
            variant_name.span()
        );
        
        match &variant.fields {
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                // 单个字段的元组变体，如 Latest(Latest)
                let field_type = &fields.unnamed.first().unwrap().ty;
                quote! {
                    #[classmethod]
                    fn #method_name(_cls: &::pyo3::Bound<'_, ::pyo3::types::PyType>, value: #field_type) -> Self {
                        Self::#variant_name(value)
                    }
                }
            }
            Fields::Named(_) => {
                // 命名字段（通常不会出现在 protobuf oneof 中）
                quote! {}
            }
            Fields::Unit => {
                // 无字段变体
                quote! {
                    #[staticmethod]
                    fn #method_name() -> Self {
                        Self::#variant_name
                    }
                }
            }
            _ => quote! {}
        }
    });

    let expanded = quote! {
        #[automatically_derived]
        const _: () = {
            use ::pyo3::prelude::*;

            // 为 oneof enum 的每个变体生成构造方法
            #[::pyo3::pymethods]
            impl #name {
                #(#variant_constructors)*
            }
        };
    };

    TokenStream::from(expanded)
}

/// 自动为包含 Vec<T> 字段的类型生成零拷贝方法
///
/// 这个宏会为类型中的每个 `Vec<T>` 字段（T 为数值类型）生成 `replace_*` 方法，
/// 这些方法可以：
/// - 从 numpy 数组零拷贝设置数据
/// - 返回 numpy 数组（零拷贝）
///
/// # 示例
///
/// ```rust
/// #[derive(ImplPyZeroCopy)]
/// struct MyStruct {
///     pub fp32_contents: Vec<f32>,
///     pub int_contents: Vec<i32>,
/// }
/// ```
///
/// 会生成：
/// - `replace_fp32_contents()` - 设置/获取 f32 数组
/// - `replace_int_contents()` - 设置/获取 i32 数组
#[proc_macro_derive(ImplPyZeroCopy)]
pub fn impl_py_zero_copy(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    
    let name = &input.ident;
    
    match &input.data {
        Data::Struct(data_struct) => generate_zero_copy_impl(name, &data_struct.fields),
        _ => {
            TokenStream::from(quote! {})
        }
    }
}

/// 自动为包含 Vec<T> 字段的类型生成 getter/setter 方法
///
/// 这个宏会为类型中的每个 `Vec<T>` 字段生成：
/// - getter: 返回 `VecRef<T>`（只读视图）
/// - setter: 接受 `List<T>`（可变容器）
///
/// # 示例
///
/// ```rust
/// #[derive(ImplPyVecAccessors)]
/// struct MyStruct {
///     pub fp32_contents: Vec<f32>,
/// }
/// ```
///
/// 会生成：
/// - `get_fp32_contents()` - 返回 VecRef<f32>
/// - `set_fp32_contents(list: List<f32>)` - 设置数据
#[proc_macro_derive(ImplPyVecAccessors)]
pub fn impl_py_vec_accessors(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    
    let name = &input.ident;
    
    match &input.data {
        Data::Struct(data_struct) => generate_vec_accessors_impl(name, &data_struct.fields),
        _ => {
            TokenStream::from(quote! {})
        }
    }
}

fn generate_zero_copy_impl(name: &syn::Ident, fields: &Fields) -> TokenStream {
    let field_info: Vec<_> = match fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .filter_map(|f| {
                let field_name = f.ident.as_ref()?;
                let field_type = &f.ty;
                
                // 检查是否是 Vec<T> 类型
                if let Some(inner_type) = extract_vec_inner_type(field_type) {
                    // 检查是否是嵌套的 Vec<Vec<u8>>（bytes_contents）
                    if let Some(nested_inner) = extract_vec_inner_type(&inner_type) {
                        if is_supported_numeric_type(&nested_inner) {
                            // 这是 Vec<Vec<T>>，需要特殊处理
                            return Some((field_name.clone(), inner_type, true));
                        }
                    }
                    
                    // 检查是否是支持的数值类型
                    if is_supported_numeric_type(&inner_type) {
                        Some((field_name.clone(), inner_type, false))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect(),
        _ => vec![],
    };
    
    // 为每个 Vec<T> 字段生成 replace 方法
    let replace_methods: Vec<_> = field_info.iter().map(|(field_name, inner_type, is_nested)| {
        let method_name = syn::Ident::new(
            &format!("replace_{}", field_name),
            field_name.span(),
        );
        
        if *is_nested {
            // 处理 Vec<Vec<u8>> 类型（bytes_contents）
            // 需要特殊处理，使用 PyArray2
            quote! {
                #[pyo3(signature = (src=None))]
                fn #method_name<'py>(
                    &mut self,
                    py: Python<'py>,
                    src: Option<::numpy::PyReadonlyArray2<u8>>,
                ) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::numpy::PyArray2<u8>>> {
                    use ::numpy::{PyArrayMethods, PyUntypedArrayMethods};
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
                    let contents = ::std::mem::replace(&mut self.#field_name, src);
                    Ok(::numpy::PyArray2::from_vec2(py, contents.as_slice())?)
                }
            }
        } else {
            // 处理普通的 Vec<T> 类型
            let (py_array_type, py_readonly_type) = get_numpy_types(inner_type);
            
            quote! {
                #[pyo3(signature = (src=None))]
                fn #method_name<'py>(
                    &mut self,
                    py: Python<'py>,
                    src: Option<#py_readonly_type>,
                ) -> ::pyo3::PyResult<::pyo3::Bound<'py, #py_array_type>> {
                    use ::numpy::{PyArrayMethods, PyUntypedArrayMethods};
                    let src = match src {
                        None => {
                            vec![]
                        }
                        Some(src) => src.to_vec()?,
                    };
                    let contents = ::std::mem::replace(&mut self.#field_name, src);
                    Ok(#py_array_type::from_vec(py, contents))
                }
            }
        }
    }).collect();
    
    if replace_methods.is_empty() {
        return TokenStream::from(quote! {});
    }
    
    let expanded = quote! {
        #[automatically_derived]
        const _: () = {
            use ::pyo3::prelude::*;
            use ::numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
            
            #[::pyo3::pymethods]
            impl #name {
                #(#replace_methods)*
            }
        };
    };
    
    TokenStream::from(expanded)
}

/// 从类型中提取 Vec<T> 的 T
fn extract_vec_inner_type(ty: &Type) -> Option<Type> {
    if let Type::Path(type_path) = ty {
        let segments = &type_path.path.segments;
        
        // 检查是否是 Vec<T>（标准库）
        if segments.len() == 1 {
            if let Some(segment) = segments.last() {
                if segment.ident == "Vec" {
                    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                            return Some(inner.clone());
                        }
                    }
                }
            }
        }
        
        // 检查是否是 prost::alloc::vec::Vec<T>
        // 路径应该是: prost -> alloc -> vec -> Vec<T>
        if segments.len() >= 3 {
            if let Some(segment) = segments.last() {
                if segment.ident == "Vec" {
                    // 检查前面的路径段
                    let prev_segments: Vec<_> = segments.iter().take(segments.len() - 1).collect();
                    if prev_segments.len() >= 2 {
                        let last_two: Vec<_> = prev_segments.iter().rev().take(2).collect();
                        if last_two.len() == 2 {
                            if last_two[0].ident == "vec" && last_two[1].ident == "alloc" {
                                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                                        return Some(inner.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// 检查是否是支持的数值类型
fn is_supported_numeric_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            let type_name = segment.ident.to_string();
            matches!(
                type_name.as_str(),
                "bool" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "f32" | "f64"
            )
        } else {
            false
        }
    } else {
        false
    }
}

/// 根据 Rust 类型获取对应的 numpy 类型
fn get_numpy_types(ty: &Type) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            let type_name = segment.ident.to_string();
            match type_name.as_str() {
                "bool" => (
                    quote! { ::numpy::PyArray1<bool> },
                    quote! { ::numpy::PyReadonlyArray1<bool> },
                ),
                "i8" => (
                    quote! { ::numpy::PyArray1<i8> },
                    quote! { ::numpy::PyReadonlyArray1<i8> },
                ),
                "i16" => (
                    quote! { ::numpy::PyArray1<i16> },
                    quote! { ::numpy::PyReadonlyArray1<i16> },
                ),
                "i32" => (
                    quote! { ::numpy::PyArray1<i32> },
                    quote! { ::numpy::PyReadonlyArray1<i32> },
                ),
                "i64" => (
                    quote! { ::numpy::PyArray1<i64> },
                    quote! { ::numpy::PyReadonlyArray1<i64> },
                ),
                "u8" => (
                    quote! { ::numpy::PyArray1<u8> },
                    quote! { ::numpy::PyReadonlyArray1<u8> },
                ),
                "u16" => (
                    quote! { ::numpy::PyArray1<u16> },
                    quote! { ::numpy::PyReadonlyArray1<u16> },
                ),
                "u32" => (
                    quote! { ::numpy::PyArray1<u32> },
                    quote! { ::numpy::PyReadonlyArray1<u32> },
                ),
                "u64" => (
                    quote! { ::numpy::PyArray1<u64> },
                    quote! { ::numpy::PyReadonlyArray1<u64> },
                ),
                "f32" => (
                    quote! { ::numpy::PyArray1<f32> },
                    quote! { ::numpy::PyReadonlyArray1<f32> },
                ),
                "f64" => (
                    quote! { ::numpy::PyArray1<f64> },
                    quote! { ::numpy::PyReadonlyArray1<f64> },
                ),
                _ => {
                    // 默认使用 i32（不应该到达这里）
                    (
                        quote! { ::numpy::PyArray1<i32> },
                        quote! { ::numpy::PyReadonlyArray1<i32> },
                    )
                }
            }
        } else {
            (
                quote! { ::numpy::PyArray1<i32> },
                quote! { ::numpy::PyReadonlyArray1<i32> },
            )
        }
    } else {
        (
            quote! { ::numpy::PyArray1<i32> },
            quote! { ::numpy::PyReadonlyArray1<i32> },
        )
    }
}

/// 生成 Vec 字段的 getter/setter
fn generate_vec_accessors_impl(name: &syn::Ident, fields: &Fields) -> TokenStream {
    let field_info: Vec<_> = match fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .filter_map(|f| {
                let field_name = f.ident.as_ref()?;
                let field_type = &f.ty;
                
                // 检查是否是 Vec<T> 类型
                if let Some(inner_type) = extract_vec_inner_type(field_type) {
                    // 检查是否是支持的数值类型
                    if is_supported_numeric_type(&inner_type) {
                        Some((field_name.clone(), inner_type))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect(),
        _ => vec![],
    };
    
    // 为每个 Vec<T> 字段生成 getter/setter
    let accessors: Vec<_> = field_info.iter().map(|(field_name, inner_type)| {
        // let field_name_str = capitalize_first_letter(field_name.to_string());
        let getter_name = syn::Ident::new(
            &format!("Get_{}", field_name),
            field_name.span(),
        );
        let setter_name = syn::Ident::new(
            &format!("Set_{}", field_name),
            field_name.span(),
        );

        let take_name = syn::Ident::new(
            &format!("Take_{}", field_name),
            field_name.span(),
        );

        let repace_name = syn::Ident::new(
            &format!("Replace_{}", field_name),
            field_name.span(),
        );


        let list_type_name = get_type_info(inner_type);
        
        quote! {
            // get 方法：简单克隆，用于一般访问（性能不敏感）
            #[allow(non_snake_case)]
            fn #getter_name(&self, py: Python) -> ::py_vec_types::#list_type_name {
                ::py_vec_types::#list_type_name::new(self.#field_name.clone())
            }

            // take 方法：转移所有权，用于 model_infer 等性能关键场景
            #[allow(non_snake_case)]
            fn #take_name(&mut self, py: Python) -> ::py_vec_types::#list_type_name {
                let data = std::mem::replace(&mut self.#field_name, vec![]);
                ::py_vec_types::#list_type_name::new(data)
            }

            // replace 方法：转移所有权替换，用于 model_infer 等性能关键场景
            // 通过 extract 实现所有权转移，避免数据拷贝
            #[allow(non_snake_case)]
            fn #repace_name(&mut self, py: Python, val: ::pyo3::Bound<'_, ::pyo3::PyAny>) -> pyo3::PyResult<::py_vec_types::#list_type_name> {
                let new_val: ::py_vec_types::#list_type_name = pyo3::types::PyAnyMethods::extract(&val)?;
                let data = std::mem::replace(&mut self.#field_name, new_val.into_vec());
                Ok(::py_vec_types::#list_type_name::new(data))
            }
            
            // set 方法：简单设置，直接拷贝数据（性能不敏感的场景）
            // 直接接受 List 类型，不需要 extract，因为普通设置操作拷贝数据也可以接受
            #[allow(non_snake_case)]
            fn #setter_name(&mut self, list: ::py_vec_types::#list_type_name) {
                self.#field_name = list.into_vec();
            }
        }
    }).collect();
    
    if accessors.is_empty() {
        return TokenStream::from(quote! {});
    }
    
    let expanded = quote! {
        #[automatically_derived]
        const _: () = {
            use ::pyo3::prelude::*;
            use ::py_vec_types;
            
            #[::pyo3::pymethods]
            impl #name {
                #(#accessors)*
            }
        };
    };
    
    TokenStream::from(expanded)
}

/// 获取类型信息（PyArray 类型和 List/VecRef 类型名）
fn get_type_info(ty: &Type) -> syn::Ident {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            let type_name = segment.ident.to_string();
            let list_name = match type_name.as_str() {
                "bool" => (
                    syn::Ident::new("ListBool", segment.ident.span()),
                ),
                "i8" => (
                    syn::Ident::new("ListI8", segment.ident.span()),
                ),
                "i16" => (
                    syn::Ident::new("ListI16", segment.ident.span()),
                ),
                "i32" => (
                    syn::Ident::new("ListI32", segment.ident.span()),
                ),
                "i64" => (
                    syn::Ident::new("ListI64", segment.ident.span()),
                ),
                "u8" => (
                    syn::Ident::new("ListU8", segment.ident.span()),
                ),
                "u16" => (
                    syn::Ident::new("ListU16", segment.ident.span()),
                ),
                "u32" => (
                    syn::Ident::new("ListU32", segment.ident.span()),
                ),
                "u64" => (
                    syn::Ident::new("ListU64", segment.ident.span()),
                ),
                "f32" => (
                    syn::Ident::new("ListF32", segment.ident.span()),
                ),
                "f64" => (
                    syn::Ident::new("ListF64", segment.ident.span()),
                ),
                _ => (
                    syn::Ident::new("ListI32", segment.ident.span()),
                ),
            };
            list_name.0
        } else {
            syn::Ident::new("ListI32", proc_macro2::Span::call_site())
        }
    } else {
        syn::Ident::new("ListI32", proc_macro2::Span::call_site())
    }
}

/// 将 PascalCase 转换为 snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch.is_uppercase() {
            if !result.is_empty() {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
        } else {
            result.push(ch);
        }
    }

    result
}

// fn capitalize_first_letter(s: String) -> String {
//     let mut chars = s.chars();
//
//     match chars.next() {
//         Some(first) if first.is_alphabetic() => {
//             let mut result = first.to_uppercase().to_string();
//             result.push_str(chars.as_str());
//             result
//         }
//         _ => s,
//     }
// }
