/// This file generates the client libraries from Triton's shared proto-buf definitions.
use anyhow::{Context, Result};
use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;


/// Return a list of all .proto files in the given directory, recursively.
fn get_protobuf_paths<P: AsRef<Path>>(directory: P) -> std::io::Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = vec![];
    for entry in WalkDir::new(directory) {
        let path = entry?.into_path();
        if path.extension() == Some(OsStr::new("proto")) {
            paths.push(path.to_path_buf());
        }
    }
    Ok(paths)
}

fn main() -> Result<()> {
    let (protoc_bin, _) = protoc_prebuilt::init("33.0").unwrap();
    unsafe { env::set_var("PROTOC", protoc_bin) };
    // The toplevel types directory.
    let pb_dir: PathBuf = env::var("TRITON_PROTOBUF")
        .ok()
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/common/protobuf").to_string())
        .into();

    println!("cargo:rerun-if-changed={}", &pb_dir.display());

    let protobuf_paths = get_protobuf_paths(&pb_dir).context(format!(
        "failed to find Protocol Buffers paths for {}",
        pb_dir.display()
    ))?;

    for path in &protobuf_paths {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    tonic_prost_build::configure()
        .build_server(true)
        .out_dir("src")
        .compile_protos(&protobuf_paths, &[pb_dir])
        .context("unable to compile Protocol Buffers for the Triton client")?;

    Ok(())
}
