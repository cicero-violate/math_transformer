use serde::{Deserialize, Serialize};

// ── Node taxonomy — complete Rust syntax space ────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    // file / module declarations
    File,
    UseDecl,
    ModDecl,
    ExternCrate,
    // top-level items
    Function,
    Struct,
    Enum,
    Trait,
    ImplBlock,
    TraitImpl,
    TypeAlias,
    Const,
    Static,
    MacroDef,
    // members
    StructField,
    EnumVariant,
    ImplItem,
    TraitItem,
    // signature parts
    GenericParam,
    Lifetime,
    WhereClause,
    FnParam,
    ReturnType,
    // body nodes
    MatchArm,
    Expr,
    Stmt,
    Block,
    // meta
    Attribute,
    DocComment,
    // type system
    TypeRef,
    Pattern,
    // macro invocation
    MacroCall,
}

// ── Node locator ──────────────────────────────────────────────────────────────

/// How to address a node. Anchor is byte-precise (from compiler index);
/// Selector is a best-effort structural path used when no compiler data is available.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "loc", rename_all = "snake_case")]
pub enum NodeLocator {
    /// Byte-precise span from the compiler semantic index.
    Anchor {
        path: String,
        byte_from: usize,
        byte_to: usize,
    },
    /// Structural selector, e.g. "struct Config::field enabled",
    /// "impl Config::fn new", "enum Status::variant Active".
    Selector { path: String, selector: String },
}

impl NodeLocator {
    pub fn path(&self) -> &str {
        match self {
            NodeLocator::Anchor { path, .. } => path,
            NodeLocator::Selector { path, .. } => path,
        }
    }
}

// ── What part of a node to replace ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplaceTarget {
    Whole,     // entire node text
    Body,      // content between braces / after `=`
    Signature, // name + params + return type (before body)
    Type,      // type annotation only
    Value,     // initializer / default value
}

// ── Edge kinds — syntactic relationships ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "edge", rename_all = "snake_case")]
pub enum EdgeKind {
    /// `use <use_path>;`
    Uses { use_path: String },
    /// `mod <module>;` or `mod <module> { }` when `inline` is true
    Declares { module: String, inline: bool },
    /// `impl <trait_path> for <type_path> { }`
    Implements {
        trait_path: String,
        type_path: String,
    },
    /// `where <param>: <bound>` clause
    Bound { param: String, bound: String },
    /// Rewrite every `<old_path>` reference to `<new_path>` in a file
    PathRef { old_path: String, new_path: String },
    /// `extern crate <name>;`
    ExternCrate(String),
}

// ── Attribute keys ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttrKey {
    Visibility, // "pub", "pub(crate)", ""
    Derive,     // appended to / removed from derive list
    Cfg,        // cfg condition string
    Allow,      // allow lint name
    MustUse,    // "" or message
    Inline,     // "always", "never", ""
    Deprecated, // "" or message
    Doc,        // doc comment text
    Repr,       // repr string e.g. "C", "u8"
    Custom,     // raw attribute text
}

// ── Verify predicates ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "predicate", rename_all = "snake_case")]
pub enum VerifyPredicate {
    FileExists { path: String },
    FileAbsent { path: String },
    ContainsText { path: String, text: String },
    TextAbsent { path: String, text: String },
    SymbolExists { path: String, selector: String },
}

// ── Basis ops (B) ─────────────────────────────────────────────────────────────

/// The structural basis — every refactoring decomposes into a sequence of these.
/// Extract and Inline are NOT ops; they are composite sequences at a higher layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum StructuralOp {
    /// Add a new node of `kind` at `at`, rendered from `text`.
    CreateNode(CreateNode),
    /// Remove the node addressed by `at`.
    DeleteNode(DeleteNode),
    /// Replace `target` part of the node at `at` with `text`.
    ReplaceNode(ReplaceNode),
    /// Move the node at `from` to `to`; optionally write a re-export facade.
    MoveNode(MoveNode),
    /// Rename a symbol at `at` to `new_name` across `scope` files.
    RenameSymbol(RenameSymbol),
    /// Set a structured attribute on the node at `at`.
    SetAttr(SetAttr),
    /// Add a syntactic edge (use, mod, impl, where bound, path ref) to `file`.
    AddEdge(AddEdge),
    /// Remove a syntactic edge from `file`.
    RemoveEdge(RemoveEdge),
    /// Assert a predicate holds; fail the batch if it does not.
    Verify(Verify),
    // ── meta ops (manifest / audit trail) ───────────────────────────────────
    Cargo(CargoChange),
    Receipt(Receipt),
    Rollback(Rollback),
}

// ── CreateNode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateNode {
    pub kind: NodeKind,
    /// Where to insert. For files this is ignored (path in locator is created).
    pub at: NodeLocator,
    /// Fully rendered text of the new node (including attributes, visibility, body).
    pub text: String,
}

// ── DeleteNode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteNode {
    pub kind: NodeKind,
    pub at: NodeLocator,
    /// Guard: text-addressed item deletion requires compiler proof of non-use.
    #[serde(default)]
    pub compiler_proven_unused: bool,
}

// ── ReplaceNode ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaceNode {
    pub kind: NodeKind,
    pub at: NodeLocator,
    pub target: ReplaceTarget,
    pub text: String,
}

// ── MoveNode ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveNode {
    pub kind: NodeKind,
    pub from: NodeLocator,
    pub to: NodeLocator,
    /// Write a re-export facade at the source after moving.
    #[serde(default)]
    pub preserve_facade: bool,
    pub facade_text: Option<String>,
}

// ── RenameSymbol ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenameSymbol {
    pub kind: NodeKind,
    /// Location of the definition being renamed.
    pub at: NodeLocator,
    pub old_name: String,
    pub new_name: String,
    /// Files to search for references; definition file is always included.
    #[serde(default)]
    pub scope: Vec<String>,
}

// ── SetAttr ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetAttr {
    pub kind: NodeKind,
    pub at: NodeLocator,
    pub key: AttrKey,
    /// New value. Empty string removes the attribute.
    pub value: String,
}

// ── AddEdge / RemoveEdge ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddEdge {
    /// File that receives the edge (e.g. the use/mod/where goes here).
    pub file: String,
    pub edge: EdgeKind,
    /// Insert near this locator instead of at the top of the file.
    pub near: Option<NodeLocator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveEdge {
    pub file: String,
    pub edge: EdgeKind,
}

// ── Verify ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verify {
    pub predicate: VerifyPredicate,
    /// Human-readable message emitted on failure.
    pub message: Option<String>,
}

// ── Cargo ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "change", rename_all = "snake_case")]
pub enum CargoChange {
    AddDependency {
        manifest: String,
        name: String,
        version: String,
        #[serde(default)]
        features: Vec<String>,
    },
    RemoveDependency {
        manifest: String,
        name: String,
    },
    AddDevDependency {
        manifest: String,
        name: String,
        version: String,
        #[serde(default)]
        features: Vec<String>,
    },
    RemoveDevDependency {
        manifest: String,
        name: String,
    },
    AddBuildDependency {
        manifest: String,
        name: String,
        version: String,
        #[serde(default)]
        features: Vec<String>,
    },
    AddFeature {
        manifest: String,
        name: String,
        #[serde(default)]
        members: Vec<String>,
    },
    RemoveFeature {
        manifest: String,
        name: String,
    },
    SetPackageField {
        manifest: String,
        field: String,
        value: String,
    },
    AddBinTarget {
        manifest: String,
        name: String,
        path: String,
    },
    AddLibTarget {
        manifest: String,
        path: String,
        crate_type: Option<String>,
    },
    AddTestTarget {
        manifest: String,
        name: String,
        path: String,
    },
    AddExampleTarget {
        manifest: String,
        name: String,
        path: String,
    },
    RemoveTarget {
        manifest: String,
        kind: String,
        name: String,
    },
    InsertSnippet {
        manifest: String,
        snippet: String,
    },
}

// ── Meta ops ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    pub summary: String,
    pub rollback_required: bool,
    pub receipt_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rollback {
    pub manifest: String,
    pub rollback_path: String,
}

// ── Batch ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpBatch {
    pub label: Option<String>,
    pub ops: Vec<StructuralOp>,
}
