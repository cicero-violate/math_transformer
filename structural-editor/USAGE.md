# structural-editor usage

A JSONL stdin → stdout process. Send one JSON object per line; receive one result line back.

## Build

```bash
cd /workspace/ai_sandbox/canon-mini-agent/prototype
cargo build -p structural-editor
# binary: target/debug/structural-editor
```

## Wire format

**Input line**
```json
{
  "root": "/absolute/path/to/project",
  "label": "optional-trace-label",
  "ops": [ ...operations... ]
}
```

**Output line**
```json
{
  "ok": true,
  "label": "optional-trace-label",
  "deltas": [
    { "path": "src/foo.rs", "action": "created", "diff": "--- a/src/foo.rs\n..." }
  ],
  "error": null
}
```

All ops in a batch are buffered in memory. Files are written to disk only when **every op succeeds**. On failure, `ok` is `false`, `error` contains the message, and no files are changed.

`action` is one of: `created` | `modified` | `deleted` | `moved`.

---

## Op basis

The op set is a structural basis — every refactoring decomposes into a sequence of these atoms. Extract and Inline are **not** ops; they are composite sequences encoded as `create_node` + `replace_node` + `delete_node`.

| Op | What it does |
|----|-------------|
| `create_node` | Add a new node (file, item, member, fragment) at a locator |
| `delete_node` | Remove a node at a locator |
| `replace_node` | Replace whole/body/signature/type/value of a node |
| `move_node` | Move a node from one locator to another |
| `rename_symbol` | Rename a definition and all references across a scope |
| `set_attr` | Set visibility, derive, cfg, doc, inline, etc. on a node |
| `add_edge` | Add a `use`, `mod`, `impl Trait for Type`, `where` bound, or path ref |
| `remove_edge` | Remove any of the above edges |
| `verify` | Assert a predicate — fails the batch if false |
| `cargo` | Edit a `Cargo.toml` manifest |
| `receipt` | Write an execution receipt artifact |
| `rollback` | Write a rollback manifest artifact |

---

## Node locator

Every node op addresses its target with a `NodeLocator`, which is either:

**Anchor** — byte-precise, from the compiler semantic index:
```json
{ "loc": "anchor", "path": "src/lib.rs", "byte_from": 42, "byte_to": 120 }
```

**Selector** — structural path, best-effort text resolution:
```json
{ "loc": "selector", "path": "src/lib.rs", "selector": "struct Config::field enabled" }
```

Selector syntax examples:
- `"struct Config"` — the Config struct
- `"struct Config::field enabled"` — the `enabled` field inside Config
- `"impl Config::fn new"` — the `new` method inside `impl Config`
- `"enum Status::variant Active"` — the Active variant
- `"fn process"` — a top-level function

Use `anchor` when you have compiler span data. Use `selector` when you don't.

---

## Node kinds

The complete Rust syntax space covered by `NodeKind`:

| Category | Kinds |
|----------|-------|
| File / module | `file`, `use_decl`, `mod_decl`, `extern_crate` |
| Items | `function`, `struct`, `enum`, `trait`, `impl_block`, `trait_impl`, `type_alias`, `const`, `static`, `macro_def` |
| Members | `struct_field`, `enum_variant`, `impl_item`, `trait_item` |
| Signature parts | `generic_param`, `lifetime`, `where_clause`, `fn_param`, `return_type` |
| Body nodes | `match_arm`, `expr`, `stmt`, `block` |
| Meta | `attribute`, `doc_comment` |
| Type system | `type_ref`, `pattern` |
| Macro | `macro_call` |

---

## Operations

All examples show only the op object that goes inside `"ops": [...]`.

---

### create_node

Add a new node. For `file`, creates the file with `text` as content. For all other kinds, inserts `text` at the locator position (or appends to the file if the selector resolves to the file root).

**Create a file**
```json
{
  "op": "create_node",
  "kind": "file",
  "at": { "loc": "selector", "path": "src/config.rs", "selector": "file" },
  "text": "pub mod config;\n"
}
```

**Create a struct (append to file)**
```json
{
  "op": "create_node",
  "kind": "struct",
  "at": { "loc": "selector", "path": "src/config.rs", "selector": "file" },
  "text": "#[derive(Debug, Clone)]\npub struct Config {\n    pub enabled: bool,\n    pub timeout_ms: u64,\n}\n"
}
```

**Create a field inside an existing struct (at byte anchor)**
```json
{
  "op": "create_node",
  "kind": "struct_field",
  "at": { "loc": "anchor", "path": "src/config.rs", "byte_from": 95, "byte_to": 95 },
  "text": "    pub retries: u32,\n"
}
```

**Create an enum variant**
```json
{
  "op": "create_node",
  "kind": "enum_variant",
  "at": { "loc": "selector", "path": "src/status.rs", "selector": "enum Status" },
  "text": "    Pending,\n"
}
```

**Create a function**
```json
{
  "op": "create_node",
  "kind": "function",
  "at": { "loc": "selector", "path": "src/lib.rs", "selector": "file" },
  "text": "pub fn process(input: &str) -> String {\n    input.to_uppercase()\n}\n"
}
```

**Create a where clause**
```json
{
  "op": "create_node",
  "kind": "where_clause",
  "at": { "loc": "selector", "path": "src/lib.rs", "selector": "fn process" },
  "text": "    T: Clone + Send,\n"
}
```

---

### delete_node

Remove a node. File deletion needs no guard. Item deletion via `selector` requires `compiler_proven_unused: true`; via `anchor` no guard is needed.

**Delete a file**
```json
{
  "op": "delete_node",
  "kind": "file",
  "at": { "loc": "selector", "path": "src/old.rs", "selector": "file" }
}
```

**Delete a function (anchor, no guard)**
```json
{
  "op": "delete_node",
  "kind": "function",
  "at": { "loc": "anchor", "path": "src/lib.rs", "byte_from": 42, "byte_to": 120 }
}
```

**Delete a function (selector, guarded)**
```json
{
  "op": "delete_node",
  "kind": "function",
  "at": { "loc": "selector", "path": "src/lib.rs", "selector": "fn dead_code" },
  "compiler_proven_unused": true
}
```

**Delete a struct field**
```json
{
  "op": "delete_node",
  "kind": "struct_field",
  "at": { "loc": "anchor", "path": "src/config.rs", "byte_from": 60, "byte_to": 82 }
}
```

---

### replace_node

Replace part of a node. `target` controls what is replaced:

| target | Replaces |
|--------|---------|
| `whole` | Entire node text |
| `body` | Content between `{` and `}` (or after `=`) |
| `signature` | Name + params + return type (before body) |
| `type` | Type annotation only |
| `value` | Initializer / default value |

**Replace a function body**
```json
{
  "op": "replace_node",
  "kind": "function",
  "at": { "loc": "selector", "path": "src/lib.rs", "selector": "fn process" },
  "target": "body",
  "text": "    input.trim().to_uppercase()"
}
```

**Replace a function signature**
```json
{
  "op": "replace_node",
  "kind": "function",
  "at": { "loc": "anchor", "path": "src/lib.rs", "byte_from": 0, "byte_to": 18 },
  "target": "signature",
  "text": "process(input: &str) -> String"
}
```

**Replace a field's type**
```json
{
  "op": "replace_node",
  "kind": "struct_field",
  "at": { "loc": "selector", "path": "src/config.rs", "selector": "struct Config::field timeout_ms" },
  "target": "type",
  "text": "std::time::Duration"
}
```

**Replace a whole const**
```json
{
  "op": "replace_node",
  "kind": "const",
  "at": { "loc": "selector", "path": "src/lib.rs", "selector": "MAX_RETRIES" },
  "target": "whole",
  "text": "pub const MAX_RETRIES: u32 = 5;\n"
}
```

---

### move_node

Move a node from one locator to another. For `file`, moves the file on disk. For items, cuts from source and pastes at destination.

**Move a file (with re-export facade)**
```json
{
  "op": "move_node",
  "kind": "file",
  "from": { "loc": "selector", "path": "src/old.rs", "selector": "file" },
  "to": { "loc": "selector", "path": "src/new.rs", "selector": "file" },
  "preserve_facade": true,
  "facade_text": "pub use crate::new::*;\n"
}
```

**Move a function to another file**
```json
{
  "op": "move_node",
  "kind": "function",
  "from": { "loc": "anchor", "path": "src/lib.rs", "byte_from": 100, "byte_to": 200 },
  "to": { "loc": "selector", "path": "src/utils.rs", "selector": "file" }
}
```

Note: after a `move_node`, use `add_edge` / `remove_edge` to update `use` declarations, and `rename_symbol` or `replace_node` to fix path references.

---

### rename_symbol

Rename a definition and every reference to it across the listed scope files. The definition file is always included.

```json
{
  "op": "rename_symbol",
  "kind": "struct",
  "at": { "loc": "selector", "path": "src/config.rs", "selector": "struct Config" },
  "old_name": "Config",
  "new_name": "AppConfig",
  "scope": ["src/lib.rs", "src/main.rs", "src/server.rs"]
}
```

```json
{
  "op": "rename_symbol",
  "kind": "function",
  "at": { "loc": "anchor", "path": "src/lib.rs", "byte_from": 0, "byte_to": 50 },
  "old_name": "process",
  "new_name": "handle",
  "scope": ["src/main.rs"]
}
```

---

### set_attr

Set a structured attribute on a node. An empty `value` removes the attribute.

| key | Effect |
|-----|--------|
| `visibility` | Sets `pub`, `pub(crate)`, `pub(super)`, or `` (private) |
| `derive` | Appends to (or creates) the `#[derive(...)]` list |
| `cfg` | Inserts `#[cfg(<value>)]` |
| `allow` | Inserts `#[allow(<value>)]` |
| `must_use` | Inserts `#[must_use]` or `#[must_use = "..."]` |
| `inline` | Inserts `#[inline(<value>)]` |
| `deprecated` | Inserts `#[deprecated]` or `#[deprecated = "..."]` |
| `doc` | Inserts `/// <value>` |
| `repr` | Inserts `#[repr(<value>)]` |
| `custom` | Inserts `value` verbatim as an attribute line |

**Make a struct public**
```json
{
  "op": "set_attr",
  "kind": "struct",
  "at": { "loc": "selector", "path": "src/config.rs", "selector": "struct Config" },
  "key": "visibility",
  "value": "pub"
}
```

**Add a derive**
```json
{
  "op": "set_attr",
  "kind": "struct",
  "at": { "loc": "selector", "path": "src/config.rs", "selector": "struct Config" },
  "key": "derive",
  "value": "Serialize, Deserialize"
}
```

**Add a doc comment**
```json
{
  "op": "set_attr",
  "kind": "function",
  "at": { "loc": "selector", "path": "src/lib.rs", "selector": "fn process" },
  "key": "doc",
  "value": "Process the input and return an uppercased result."
}
```

**Mark as deprecated**
```json
{
  "op": "set_attr",
  "kind": "function",
  "at": { "loc": "selector", "path": "src/lib.rs", "selector": "fn old_api" },
  "key": "deprecated",
  "value": "use process() instead"
}
```

---

### add_edge / remove_edge

Add or remove a syntactic relationship in a file.

**Add a `use` declaration**
```json
{
  "op": "add_edge",
  "file": "src/main.rs",
  "edge": { "edge": "uses", "use_path": "crate::config::AppConfig" }
}
```

**Remove a `use` declaration**
```json
{
  "op": "remove_edge",
  "file": "src/main.rs",
  "edge": { "edge": "uses", "use_path": "crate::config::Config" }
}
```

**Add a `mod` declaration**
```json
{
  "op": "add_edge",
  "file": "src/lib.rs",
  "edge": { "edge": "declares", "module": "config", "inline": false }
}
```

**Add an `impl Trait for Type` block**
```json
{
  "op": "add_edge",
  "file": "src/config.rs",
  "edge": { "edge": "implements", "trait_path": "Default", "type_path": "AppConfig" }
}
```

**Add a `where` bound**
```json
{
  "op": "add_edge",
  "file": "src/lib.rs",
  "edge": { "edge": "bound", "param": "T", "bound": "Clone + Send + 'static" }
}
```

**Rewrite all path references in a file**
```json
{
  "op": "add_edge",
  "file": "src/main.rs",
  "edge": { "edge": "path_ref", "old_path": "crate::old", "new_path": "crate::new" }
}
```

**Add `extern crate`**
```json
{
  "op": "add_edge",
  "file": "src/lib.rs",
  "edge": { "edge": "extern_crate", "name": "serde_derive" }
}
```

---

### verify

Assert a predicate before committing. If the predicate is false the entire batch fails and no files are written.

| predicate | Passes when |
|-----------|------------|
| `file_exists` | `path` exists (or was created earlier in the batch) |
| `file_absent` | `path` does not exist |
| `contains_text` | File content includes `text` |
| `text_absent` | File content does not include `text` |
| `symbol_exists` | File content includes the selector's item name |

```json
{ "op": "verify", "predicate": { "predicate": "file_exists", "path": "src/config.rs" } }
```

```json
{
  "op": "verify",
  "predicate": { "predicate": "symbol_exists", "path": "src/lib.rs", "selector": "fn process" },
  "message": "process() must exist before we can rename it"
}
```

```json
{
  "op": "verify",
  "predicate": { "predicate": "text_absent", "path": "src/lib.rs", "text": "todo!()" },
  "message": "no todo!() stubs allowed"
}
```

---

### cargo

Edit a `Cargo.toml` manifest.

```json
{ "op": "cargo", "change": "add_dependency", "manifest": "Cargo.toml",
  "name": "serde", "version": "1", "features": ["derive"] }

{ "op": "cargo", "change": "remove_dependency", "manifest": "Cargo.toml", "name": "serde" }

{ "op": "cargo", "change": "add_dev_dependency", "manifest": "Cargo.toml",
  "name": "pretty_assertions", "version": "1" }

{ "op": "cargo", "change": "add_build_dependency", "manifest": "Cargo.toml",
  "name": "tonic-build", "version": "0.11" }

{ "op": "cargo", "change": "set_package_field", "manifest": "Cargo.toml",
  "field": "edition", "value": "2021" }

{ "op": "cargo", "change": "add_feature", "manifest": "Cargo.toml",
  "name": "full", "members": ["serde", "tokio"] }

{ "op": "cargo", "change": "remove_feature", "manifest": "Cargo.toml", "name": "legacy" }

{ "op": "cargo", "change": "add_bin_target", "manifest": "Cargo.toml",
  "name": "my-tool", "path": "src/bin/my_tool.rs" }

{ "op": "cargo", "change": "add_test_target", "manifest": "Cargo.toml",
  "name": "integration", "path": "tests/integration.rs" }

{ "op": "cargo", "change": "add_example_target", "manifest": "Cargo.toml",
  "name": "demo", "path": "examples/demo.rs" }

{ "op": "cargo", "change": "add_lib_target", "manifest": "Cargo.toml",
  "path": "src/lib.rs", "crate_type": "cdylib" }

{ "op": "cargo", "change": "remove_target", "manifest": "Cargo.toml",
  "kind": "bin", "name": "old-tool" }

{ "op": "cargo", "change": "insert_snippet", "manifest": "Cargo.toml",
  "snippet": "[profile.release]\nopt-level = 3\n" }
```

---

### receipt / rollback

Write audit trail artifacts. These do not edit source files.

```json
{
  "op": "receipt",
  "summary": "renamed Config → AppConfig across 3 files",
  "rollback_required": false,
  "receipt_path": "state/receipts/rename.json"
}
```

```json
{
  "op": "rollback",
  "manifest": "{\"ops\":[...inverse ops...]}",
  "rollback_path": "state/receipts/rollback.json"
}
```

---

## Composing complex refactors

Extract and Inline are not ops. Encode them as atom sequences:

**Extract a function**
```
1. create_node(function, dest_file)   — write the new function with extracted body
2. replace_node(expr, source_anchor)  — replace extracted range with a call
3. add_edge(use, ...)                 — add use declaration if moving to another file
```

**Inline a function**
```
1. replace_node(expr, call_site_1)    — substitute each call site with the body
2. replace_node(expr, call_site_2)
3. delete_node(function, definition)  — remove the function definition
4. remove_edge(use, ...)              — remove the now-unused import
```

---

## Example batch

```bash
echo '{
  "root": "/tmp/myproject",
  "label": "add-config-struct",
  "ops": [
    {
      "op": "verify",
      "predicate": { "predicate": "file_exists", "path": "src/lib.rs" }
    },
    {
      "op": "create_node",
      "kind": "struct",
      "at": { "loc": "selector", "path": "src/lib.rs", "selector": "file" },
      "text": "#[derive(Debug, Clone)]\npub struct Config {\n    pub enabled: bool,\n}\n"
    },
    {
      "op": "set_attr",
      "kind": "struct",
      "at": { "loc": "selector", "path": "src/lib.rs", "selector": "struct Config" },
      "key": "derive",
      "value": "serde::Serialize, serde::Deserialize"
    },
    {
      "op": "cargo",
      "change": "add_dependency",
      "manifest": "Cargo.toml",
      "name": "serde",
      "version": "1",
      "features": ["derive"]
    }
  ]
}' | ./target/debug/structural-editor
```

---

## Calling from code

The process reads from stdin indefinitely. Keep it alive and pipe one batch per line.

**Python**
```python
import subprocess, json

proc = subprocess.Popen(
    ["./target/debug/structural-editor"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
)

def apply(root, ops, label=None):
    line = json.dumps({"root": root, "label": label, "ops": ops}) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())

result = apply("/tmp/myproject", [
    {
        "op": "create_node",
        "kind": "file",
        "at": {"loc": "selector", "path": "src/lib.rs", "selector": "file"},
        "text": "// generated\n"
    }
])
print(result)
```
