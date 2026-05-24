use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct FileDelta {
    pub path: String,
    pub action: FileAction,
    /// Unified diff; empty for pure deletions or artifact writes.
    pub diff: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileAction {
    Created,
    Modified,
    Deleted,
    Moved { from: String },
}

/// Structured output returned after applying one OpBatch.
#[derive(Debug, Serialize, Deserialize)]
pub struct EditResult {
    pub ok: bool,
    pub label: Option<String>,
    pub deltas: Vec<FileDelta>,
    pub error: Option<String>,
}
