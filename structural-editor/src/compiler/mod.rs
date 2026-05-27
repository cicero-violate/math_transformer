mod artifact;
mod buffer;
mod cargo_op;
mod edge;
mod node_create;
mod node_delete;
mod node_move;
mod node_replace;
mod pipeline;
mod rename;
mod set_attr;
mod text;
mod verify;

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::op::{
    AddEdge, CargoChange, CreateNode, DeleteNode, MoveNode, Receipt, RemoveEdge, RenameSymbol,
    ReplaceNode, Rollback, SetAttr, StructuralOp, Verify,
};

pub use pipeline::{OperationSource, StructuralBatchCompiler};

pub type Buffers = HashMap<PathBuf, Option<Vec<u8>>>;

/// Compiles a typed structural operation into buffered file changes.
pub trait StructuralCompiler {
    fn compile(&self, op: &StructuralOp, root: &Path, buffers: &mut Buffers) -> Result<()>;
}

/// Handler for one concrete structural operation payload.
pub trait ApplyOperation<O> {
    fn apply_operation(&self, op: &O, root: &Path, buffers: &mut Buffers) -> Result<()>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultStructuralCompiler;

impl StructuralCompiler for DefaultStructuralCompiler {
    fn compile(&self, op: &StructuralOp, root: &Path, buffers: &mut Buffers) -> Result<()> {
        match op {
            StructuralOp::CreateNode(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::DeleteNode(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::ReplaceNode(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::MoveNode(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::RenameSymbol(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::SetAttr(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::AddEdge(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::RemoveEdge(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::Verify(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::Cargo(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::Receipt(inner) => self.apply_operation(inner, root, buffers),
            StructuralOp::Rollback(inner) => self.apply_operation(inner, root, buffers),
        }
    }
}

impl ApplyOperation<CreateNode> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &CreateNode, root: &Path, buffers: &mut Buffers) -> Result<()> {
        node_create::apply(op, root, buffers)
    }
}

impl ApplyOperation<DeleteNode> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &DeleteNode, root: &Path, buffers: &mut Buffers) -> Result<()> {
        node_delete::apply(op, root, buffers)
    }
}

impl ApplyOperation<ReplaceNode> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &ReplaceNode, root: &Path, buffers: &mut Buffers) -> Result<()> {
        node_replace::apply(op, root, buffers)
    }
}

impl ApplyOperation<MoveNode> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &MoveNode, root: &Path, buffers: &mut Buffers) -> Result<()> {
        node_move::apply(op, root, buffers)
    }
}

impl ApplyOperation<RenameSymbol> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &RenameSymbol, root: &Path, buffers: &mut Buffers) -> Result<()> {
        rename::apply(op, root, buffers)
    }
}

impl ApplyOperation<SetAttr> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &SetAttr, root: &Path, buffers: &mut Buffers) -> Result<()> {
        set_attr::apply(op, root, buffers)
    }
}

impl ApplyOperation<AddEdge> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &AddEdge, root: &Path, buffers: &mut Buffers) -> Result<()> {
        edge::add(op, root, buffers)
    }
}

impl ApplyOperation<RemoveEdge> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &RemoveEdge, root: &Path, buffers: &mut Buffers) -> Result<()> {
        edge::remove(op, root, buffers)
    }
}

impl ApplyOperation<Verify> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &Verify, root: &Path, buffers: &mut Buffers) -> Result<()> {
        verify::apply(op, root, buffers)
    }
}

impl ApplyOperation<CargoChange> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &CargoChange, root: &Path, buffers: &mut Buffers) -> Result<()> {
        cargo_op::apply(op, root, buffers)
    }
}

impl ApplyOperation<Receipt> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &Receipt, root: &Path, buffers: &mut Buffers) -> Result<()> {
        artifact::apply_receipt(op, root, buffers)
    }
}

impl ApplyOperation<Rollback> for DefaultStructuralCompiler {
    fn apply_operation(&self, op: &Rollback, root: &Path, buffers: &mut Buffers) -> Result<()> {
        artifact::apply_rollback(op, root, buffers)
    }
}

/// Dispatch one StructuralOp through the default compiler implementation.
pub fn compile(op: &StructuralOp, root: &Path, buffers: &mut Buffers) -> Result<()> {
    DefaultStructuralCompiler.compile(op, root, buffers)
}
