use anyhow::{bail, Result};
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use super::buffer::load_buf;

pub type Buffers = HashMap<PathBuf, Option<Vec<u8>>>;

pub fn abs_path(root: &Path, path: &str) -> PathBuf {
    root.join(path)
}

pub fn load_text(abs: &Path, buffers: &mut Buffers) -> Result<String> {
    let bytes = load_buf(abs, buffers)?;
    Ok(bytes_to_text(&bytes))
}

pub fn load_path_bytes(
    root: &Path,
    path: &str,
    buffers: &mut Buffers,
) -> Result<(PathBuf, Vec<u8>)> {
    let abs = abs_path(root, path);
    let bytes = load_buf(&abs, buffers)?;
    Ok((abs, bytes))
}

pub fn load_path_text(root: &Path, path: &str, buffers: &mut Buffers) -> Result<(PathBuf, String)> {
    let (abs, bytes) = load_path_bytes(root, path, buffers)?;
    Ok((abs, bytes_to_text(&bytes)))
}

pub fn bytes_to_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

pub fn store_bytes(abs: PathBuf, bytes: Vec<u8>, buffers: &mut Buffers) {
    buffers.insert(abs, Some(bytes));
}

pub fn store_text(abs: PathBuf, text: String, buffers: &mut Buffers) {
    store_bytes(abs, text.into_bytes(), buffers);
}

pub fn store_path_text(root: &Path, path: &str, text: String, buffers: &mut Buffers) {
    store_text(abs_path(root, path), text, buffers);
}

pub fn delete_path(root: &Path, path: &str, buffers: &mut Buffers) {
    buffers.insert(abs_path(root, path), None);
}

pub fn splice_bytes(
    abs: PathBuf,
    buffers: &mut Buffers,
    range: Range<usize>,
    replacement: &str,
) -> Result<()> {
    edit_bytes(abs, buffers, |content| {
        content.splice(range, replacement.bytes());
    })
}

pub fn remove_bytes(abs: PathBuf, buffers: &mut Buffers, range: Range<usize>) -> Result<()> {
    edit_bytes(abs, buffers, |content| {
        content.drain(range);
    })
}

pub fn cut_bytes(abs: PathBuf, buffers: &mut Buffers, range: Range<usize>) -> Result<Vec<u8>> {
    let mut content = load_buf(&abs, buffers)?;
    let item = content[range.clone()].to_vec();
    content.drain(range);
    store_bytes(abs, content, buffers);
    Ok(item)
}

pub fn insert_bytes(abs: PathBuf, buffers: &mut Buffers, at: usize, bytes: Vec<u8>) -> Result<()> {
    edit_bytes(abs, buffers, |content| {
        let insert_at = at.min(content.len());
        content.splice(insert_at..insert_at, bytes);
    })
}

pub fn append_bytes(abs: PathBuf, buffers: &mut Buffers, bytes: &[u8]) -> Result<()> {
    edit_bytes(abs, buffers, |content| {
        content.extend_from_slice(bytes);
    })
}

pub fn buffered_exists(abs: &Path, buffers: &HashMap<PathBuf, Option<Vec<u8>>>) -> bool {
    match buffers.get(abs) {
        Some(Some(_)) => true,
        Some(None) => false,
        None => abs.exists(),
    }
}

pub fn read_buffered(abs: &Path, buffers: &HashMap<PathBuf, Option<Vec<u8>>>) -> Option<Vec<u8>> {
    match buffers.get(abs) {
        Some(Some(v)) => Some(v.clone()),
        Some(None) => None,
        None => std::fs::read(abs).ok(),
    }
}

pub fn read_buffered_text(
    abs: &Path,
    buffers: &HashMap<PathBuf, Option<Vec<u8>>>,
) -> Option<String> {
    read_buffered(abs, buffers).map(|bytes| bytes_to_text(&bytes))
}

pub fn selector_item(selector: &str) -> &str {
    selector.split("::").last().unwrap_or(selector)
}

pub fn parent_selector_item(selector: &str) -> Option<&str> {
    selector.split("::").next()
}

pub fn item_range(text: &str, selector: &str) -> Option<(usize, usize)> {
    let item = selector_item(selector);
    let start = text.find(item)?;
    let rest = &text[start..];
    let end = rest
        .find("\npub ")
        .or_else(|| rest.find("\nfn "))
        .or_else(|| rest.find("\nstruct "))
        .or_else(|| rest.find("\nimpl "))
        .unwrap_or(rest.len());
    Some((start, start + end))
}

pub fn line_start_with_indent(text: &str, item_start: usize) -> usize {
    let line_start = text[..item_start]
        .rfind('\n')
        .map(|idx| idx + 1)
        .unwrap_or(0);
    let line = &text[line_start..item_start];
    line_start + line.len() - line.trim_start().len()
}

pub fn insert_line_before(text: &str, item_start: usize, line: &str) -> String {
    let before = &text[..item_start];
    let after = &text[item_start..];
    format!("{before}{line}\n{after}")
}

pub fn filter_lines_to_string<F>(text: &str, keep: F) -> String
where
    F: Fn(&str) -> bool,
{
    text.lines()
        .filter(|line| keep(line))
        .map(|line| format!("{line}\n"))
        .collect()
}

pub fn checked_range(
    path: &str,
    byte_from: usize,
    byte_to: usize,
    len: usize,
    action: &str,
) -> Result<(usize, usize)> {
    let from = byte_from.min(len);
    let to = byte_to.min(len);
    if from > to {
        bail!("{action} anchor [{from},{to}) invalid in {path}");
    }
    Ok((from, to))
}

fn edit_bytes<F>(abs: PathBuf, buffers: &mut Buffers, edit: F) -> Result<()>
where
    F: FnOnce(&mut Vec<u8>),
{
    let mut content = load_buf(&abs, buffers)?;
    edit(&mut content);
    store_bytes(abs, content, buffers);
    Ok(())
}
