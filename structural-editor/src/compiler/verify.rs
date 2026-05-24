use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::op::{Verify, VerifyPredicate};

pub fn apply(op: &Verify, root: &Path, buffers: &HashMap<PathBuf, Option<Vec<u8>>>) -> Result<()> {
    let ok = check(&op.predicate, root, buffers);
    if !ok {
        let msg = op.message.as_deref().unwrap_or("verify predicate failed");
        bail!("{msg}: {:?}", op.predicate);
    }
    Ok(())
}

fn check(pred: &VerifyPredicate, root: &Path, buffers: &HashMap<PathBuf, Option<Vec<u8>>>) -> bool {
    match pred {
        VerifyPredicate::FileExists { path } => {
            let abs = root.join(path);
            buffered_exists(&abs, buffers)
        }
        VerifyPredicate::FileAbsent { path } => {
            let abs = root.join(path);
            !buffered_exists(&abs, buffers)
        }
        VerifyPredicate::ContainsText { path, text } => {
            let abs = root.join(path);
            read_buffered(&abs, buffers)
                .map(|bytes| String::from_utf8_lossy(&bytes).contains(text.as_str()))
                .unwrap_or(false)
        }
        VerifyPredicate::TextAbsent { path, text } => {
            let abs = root.join(path);
            read_buffered(&abs, buffers)
                .map(|bytes| !String::from_utf8_lossy(&bytes).contains(text.as_str()))
                .unwrap_or(true)
        }
        VerifyPredicate::SymbolExists { path, selector } => {
            let abs = root.join(path);
            read_buffered(&abs, buffers)
                .map(|bytes| {
                    let text = String::from_utf8_lossy(&bytes);
                    let item = selector.split("::").last().unwrap_or(selector);
                    text.contains(item)
                })
                .unwrap_or(false)
        }
    }
}

fn buffered_exists(abs: &Path, buffers: &HashMap<PathBuf, Option<Vec<u8>>>) -> bool {
    match buffers.get(abs) {
        Some(Some(_)) => true,
        Some(None) => false,
        None => abs.exists(),
    }
}

fn read_buffered(abs: &Path, buffers: &HashMap<PathBuf, Option<Vec<u8>>>) -> Option<Vec<u8>> {
    match buffers.get(abs) {
        Some(Some(v)) => Some(v.clone()),
        Some(None) => None,
        None => std::fs::read(abs).ok(),
    }
}
