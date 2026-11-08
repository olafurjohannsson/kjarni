//! `pytorch_model.bin` read without Python gives the tensors `torch.load`
//! gives: both layouts `torch.save` writes, the dtypes weights come in, views
//! that share or stride through a storage, and a wrapped state dict.
//! `fixtures/pytorch/make.py` wrote the files and the expected values.
#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};

use kjarni_transformers::tensor::DType;
use kjarni_transformers::weights::{PyTorchLoader, WeightLoader};
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pytorch")
        .join(name)
}

fn expected() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture("expected.json")).unwrap()).unwrap()
}

fn values(loader: &PyTorchLoader, name: &str) -> (Vec<usize>, Vec<f32>) {
    let view = loader.get_raw(name).unwrap();
    let values = match view.dtype {
        DType::F32 => view
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        DType::BF16 => view
            .bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect(),
        other => panic!("{name}: {other:?}"),
    };
    (view.shape.clone(), values)
}

fn check(file: &str, want: &serde_json::Map<String, Value>) {
    let loader = PyTorchLoader::new(&fixture(file)).unwrap();
    for (name, want) in want {
        if want.is_null() {
            // An integer buffer is in the file but is not a float weight.
            assert!(loader.contains(name), "{file}: {name}");
            assert!(loader.get_raw(name).is_err(), "{file}: {name}");
            continue;
        }
        let shape: Vec<usize> = serde_json::from_value(want["shape"].clone()).unwrap();
        let expect: Vec<f32> = serde_json::from_value(want["values"].clone()).unwrap();
        let (got_shape, got) = values(&loader, name);
        assert_eq!(got_shape, shape, "{file}: {name}");
        assert_eq!(got, expect, "{file}: {name}");
    }
    assert_eq!(loader.tensor_names().count(), want.len(), "{file}");
}

#[test]
fn the_zip_layout_reads_as_torch_load_does() {
    let mut want = expected().as_object().unwrap().clone();
    want.remove("nested");
    check("zip.bin", &want);
}

#[test]
fn the_legacy_layout_reads_as_torch_load_does() {
    let mut want = expected().as_object().unwrap().clone();
    want.remove("nested");
    check("legacy.bin", &want);
}

#[test]
fn a_wrapped_state_dict_reads_under_dotted_names() {
    check("nested.bin", expected()["nested"].as_object().unwrap());
}

#[test]
fn a_file_that_is_not_pytorch_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pytorch_model.bin");
    std::fs::write(&path, b"not a pickle at all").unwrap();
    assert!(PyTorchLoader::new(&path).is_err());
}

/// Every tensor of real checkpoints against `torch.load`: shape, sum, sum of
/// absolute values and first elements. Runs when `KJARNI_PYTORCH_REFERENCE`
/// names the file the reference script writes.
#[test]
fn real_checkpoints_read_as_torch_load_does() {
    let Ok(path) = std::env::var("KJARNI_PYTORCH_REFERENCE") else {
        eprintln!("skipping: KJARNI_PYTORCH_REFERENCE not set");
        return;
    };
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    for (file, tensors) in reference["files"].as_object().unwrap() {
        let loader = match PyTorchLoader::new(Path::new(file)) {
            Ok(l) => l,
            Err(e) => {
                failures.push(format!("{file}: {e:#}"));
                continue;
            }
        };
        let tensors = tensors.as_object().unwrap();
        if loader.tensor_names().count() != tensors.len() {
            failures.push(format!(
                "{file}: {} tensors, torch has {}",
                loader.tensor_names().count(),
                tensors.len()
            ));
        }
        for (name, want) in tensors {
            if want.is_null() {
                if !loader.contains(name) {
                    failures.push(format!("{file}: {name} missing"));
                }
                continue;
            }
            let (shape, got) = values(&loader, name);
            let want_shape: Vec<usize> = serde_json::from_value(want["shape"].clone()).unwrap();
            let sum: f64 = got.iter().map(|&x| x as f64).sum();
            let abs_sum: f64 = got.iter().map(|&x| (x as f64).abs()).sum();
            let first: Vec<f32> = serde_json::from_value(want["first"].clone()).unwrap();
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs().max(1.0);
            if shape != want_shape
                || !close(sum, want["sum"].as_f64().unwrap())
                || !close(abs_sum, want["abs_sum"].as_f64().unwrap())
                || got[..first.len()] != first[..]
            {
                failures.push(format!("{file}: {name} differs"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
