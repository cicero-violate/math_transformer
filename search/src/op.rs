/// An op is a raw JSON value matching structural-editor's wire format.
/// The search layer doesn't need to parse op internals — it treats ops as
/// opaque tokens proposed by the policy and executed by the sandbox.
pub type Op = serde_json::Value;

/// A batch ready to send to structural-editor over JSONL.
#[derive(serde::Serialize)]
pub struct Batch<'a> {
    pub root: &'a std::path::Path,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<&'a str>,
    pub ops: &'a [Op],
}
