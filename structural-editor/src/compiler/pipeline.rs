use anyhow::Result;
use std::path::Path;

use crate::op::StructuralOp;

use super::{Buffers, StructuralCompiler};

/// Supplies structural operations to the compiler without exposing the executor
/// to the concrete storage shape of an operation batch.
pub trait OperationSource<'a> {
    type Iter: Iterator<Item = &'a StructuralOp>;

    fn operations(&'a self) -> Self::Iter;
}

impl<'a> OperationSource<'a> for [StructuralOp] {
    type Iter = std::slice::Iter<'a, StructuralOp>;

    fn operations(&'a self) -> Self::Iter {
        self.iter()
    }
}

impl<'a> OperationSource<'a> for Vec<StructuralOp> {
    type Iter = std::slice::Iter<'a, StructuralOp>;

    fn operations(&'a self) -> Self::Iter {
        self.iter()
    }
}

/// Compiles an ordered structural operation stream into buffered file changes.
pub trait StructuralBatchCompiler {
    fn compile_batch<'a, S>(&self, ops: &'a S, root: &Path, buffers: &mut Buffers) -> Result<()>
    where
        S: OperationSource<'a> + ?Sized;
}

impl<C> StructuralBatchCompiler for C
where
    C: StructuralCompiler,
{
    fn compile_batch<'a, S>(&self, ops: &'a S, root: &Path, buffers: &mut Buffers) -> Result<()>
    where
        S: OperationSource<'a> + ?Sized,
    {
        for op in ops.operations() {
            self.compile(op, root, buffers)?;
        }
        Ok(())
    }
}
