mod artifact;
mod cargo_op;
mod edge;
mod node_create;
mod node_delete;
mod node_move;
mod node_replace;
mod rename;
mod set_attr;
mod verify;

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::op::StructuralOp;

/// Dispatch one StructuralOp to the appropriate compiler sub-module.
pub fn compile(
    op: &StructuralOp,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    match op {
        StructuralOp::CreateNode(inner) => node_create::apply(inner, root, buffers),
        StructuralOp::DeleteNode(inner) => node_delete::apply(inner, root, buffers),
        StructuralOp::ReplaceNode(inner) => node_replace::apply(inner, root, buffers),
        StructuralOp::MoveNode(inner) => node_move::apply(inner, root, buffers),
        StructuralOp::RenameSymbol(inner) => rename::apply(inner, root, buffers),
        StructuralOp::SetAttr(inner) => set_attr::apply(inner, root, buffers),
        StructuralOp::AddEdge(inner) => edge::add(inner, root, buffers),
        StructuralOp::RemoveEdge(inner) => edge::remove(inner, root, buffers),
        StructuralOp::Verify(inner) => verify::apply(inner, root, buffers),
        StructuralOp::Cargo(inner) => cargo_op::apply(inner, root, buffers),
        StructuralOp::Receipt(inner) => artifact::apply_receipt(inner, root, buffers),
        StructuralOp::Rollback(inner) => artifact::apply_rollback(inner, root, buffers),
    }
}
