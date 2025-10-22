use pyo3::prelude::*;
use triton_client_macros::ImplPyNew;

#[::pyo3::pyclass(get_all, set_all)]
#[derive(ImplPyNew)]
pub struct AAA {
    a: usize,
    b: String,
}

#[::pyo3::pyclass(get_all, set_all)]
#[derive(ImplPyNew)]
pub enum DDD {
    A(String),
    B(usize),
    C { x: usize, y: usize },
    D(Vec<usize>),
}

fn test(){

}