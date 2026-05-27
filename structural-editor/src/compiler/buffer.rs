use anyhow::{bail, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub fn load_buf(abs: &Path, buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>) -> Result<Vec<u8>> {
    if let Some(entry) = buffers.get(abs) {
        return match entry {
            Some(v) => Ok(v.clone()),
            None => bail!("file {} was deleted earlier in this batch", abs.display()),
        };
    }
    if abs.exists() {
        Ok(fs::read(abs)?)
    } else {
        Ok(Vec::new())
    }
}
