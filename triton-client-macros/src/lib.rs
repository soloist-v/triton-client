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
    let field_init = field_info.iter().map(|(name, ty)| {
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

/// 检查类型是否是 Option<T>
fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Option";
        }
    }
    false
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
