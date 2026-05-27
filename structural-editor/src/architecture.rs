//! Public architectural boundary markers for the structural editor crate.
//!
//! These traits make the pipeline's invariants explicit at the type level: each
//! phase boundary has a named capability, and the exported architecture witness
//! implements the full set. The traits are intentionally marker-style because
//! the operational behavior remains in the compiler and executor modules.

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StructuralEditorArchitecture;

pub trait TypedOperationBoundary {}
impl TypedOperationBoundary for StructuralEditorArchitecture {}

pub trait LocatorResolutionBoundary {}
impl LocatorResolutionBoundary for StructuralEditorArchitecture {}

pub trait TextSelectionBoundary {}
impl TextSelectionBoundary for StructuralEditorArchitecture {}

pub trait NodeCreationBoundary {}
impl NodeCreationBoundary for StructuralEditorArchitecture {}

pub trait NodeDeletionBoundary {}
impl NodeDeletionBoundary for StructuralEditorArchitecture {}

pub trait NodeReplacementBoundary {}
impl NodeReplacementBoundary for StructuralEditorArchitecture {}

pub trait NodeMoveBoundary {}
impl NodeMoveBoundary for StructuralEditorArchitecture {}

pub trait RenameBoundary {}
impl RenameBoundary for StructuralEditorArchitecture {}

pub trait AttributeBoundary {}
impl AttributeBoundary for StructuralEditorArchitecture {}

pub trait EdgeInsertionBoundary {}
impl EdgeInsertionBoundary for StructuralEditorArchitecture {}

pub trait EdgeRemovalBoundary {}
impl EdgeRemovalBoundary for StructuralEditorArchitecture {}

pub trait VerificationBoundary {}
impl VerificationBoundary for StructuralEditorArchitecture {}

pub trait CargoManifestBoundary {}
impl CargoManifestBoundary for StructuralEditorArchitecture {}

pub trait ReceiptBoundary {}
impl ReceiptBoundary for StructuralEditorArchitecture {}

pub trait RollbackBoundary {}
impl RollbackBoundary for StructuralEditorArchitecture {}

pub trait BufferedMutationBoundary {}
impl BufferedMutationBoundary for StructuralEditorArchitecture {}

pub trait AtomicCommitBoundary {}
impl AtomicCommitBoundary for StructuralEditorArchitecture {}

pub trait FilesystemCommitBoundary {}
impl FilesystemCommitBoundary for StructuralEditorArchitecture {}

pub trait DiffRenderingBoundary {}
impl DiffRenderingBoundary for StructuralEditorArchitecture {}

pub trait JsonInputBoundary {}
impl JsonInputBoundary for StructuralEditorArchitecture {}

pub trait JsonOutputBoundary {}
impl JsonOutputBoundary for StructuralEditorArchitecture {}

pub trait BatchLabelBoundary {}
impl BatchLabelBoundary for StructuralEditorArchitecture {}

pub trait ErrorProjectionBoundary {}
impl ErrorProjectionBoundary for StructuralEditorArchitecture {}

pub trait AllOrNothingBoundary {}
impl AllOrNothingBoundary for StructuralEditorArchitecture {}

pub trait SelectorSyntaxBoundary {}
impl SelectorSyntaxBoundary for StructuralEditorArchitecture {}

pub trait AnchorSpanBoundary {}
impl AnchorSpanBoundary for StructuralEditorArchitecture {}

pub trait RenderedTextBoundary {}
impl RenderedTextBoundary for StructuralEditorArchitecture {}

pub trait UseEdgeBoundary {}
impl UseEdgeBoundary for StructuralEditorArchitecture {}

pub trait ModuleEdgeBoundary {}
impl ModuleEdgeBoundary for StructuralEditorArchitecture {}

pub trait ImplEdgeBoundary {}
impl ImplEdgeBoundary for StructuralEditorArchitecture {}

pub trait BoundEdgeBoundary {}
impl BoundEdgeBoundary for StructuralEditorArchitecture {}

pub trait PathRewriteBoundary {}
impl PathRewriteBoundary for StructuralEditorArchitecture {}

pub trait ExternCrateBoundary {}
impl ExternCrateBoundary for StructuralEditorArchitecture {}

pub trait FilePresenceBoundary {}
impl FilePresenceBoundary for StructuralEditorArchitecture {}

pub trait TextPresenceBoundary {}
impl TextPresenceBoundary for StructuralEditorArchitecture {}

pub trait SymbolPresenceBoundary {}
impl SymbolPresenceBoundary for StructuralEditorArchitecture {}

pub trait CompilerProofBoundary {}
impl CompilerProofBoundary for StructuralEditorArchitecture {}

pub trait FacadePreservationBoundary {}
impl FacadePreservationBoundary for StructuralEditorArchitecture {}

pub trait ManifestChangeBoundary {}
impl ManifestChangeBoundary for StructuralEditorArchitecture {}

pub trait ReceiptArtifactBoundary {}
impl ReceiptArtifactBoundary for StructuralEditorArchitecture {}

pub trait RollbackArtifactBoundary {}
impl RollbackArtifactBoundary for StructuralEditorArchitecture {}

pub trait OperationStreamBoundary {}
impl OperationStreamBoundary for StructuralEditorArchitecture {}

pub trait BatchCompilationBoundary {}
impl BatchCompilationBoundary for StructuralEditorArchitecture {}

pub trait OperationDispatchBoundary {}
impl OperationDispatchBoundary for StructuralEditorArchitecture {}

pub trait CommitterBoundary {}
impl CommitterBoundary for StructuralEditorArchitecture {}

pub trait BatchExecutionBoundary {}
impl BatchExecutionBoundary for StructuralEditorArchitecture {}

pub trait EditorApiBoundary {}
impl EditorApiBoundary for StructuralEditorArchitecture {}

pub trait CliBoundary {}
impl CliBoundary for StructuralEditorArchitecture {}

pub trait OperationValidationBoundary {}
impl OperationValidationBoundary for StructuralEditorArchitecture {}

pub trait OperationNormalizationBoundary {}
impl OperationNormalizationBoundary for StructuralEditorArchitecture {}

pub trait LocatorNormalizationBoundary {}
impl LocatorNormalizationBoundary for StructuralEditorArchitecture {}

pub trait SelectorResolutionBoundary {}
impl SelectorResolutionBoundary for StructuralEditorArchitecture {}

pub trait AnchorResolutionBoundary {}
impl AnchorResolutionBoundary for StructuralEditorArchitecture {}

pub trait TextPatchPlanningBoundary {}
impl TextPatchPlanningBoundary for StructuralEditorArchitecture {}

pub trait FileDeltaProjectionBoundary {}
impl FileDeltaProjectionBoundary for StructuralEditorArchitecture {}

pub trait EditResultProjectionBoundary {}
impl EditResultProjectionBoundary for StructuralEditorArchitecture {}

pub trait ReceiptWritingBoundary {}
impl ReceiptWritingBoundary for StructuralEditorArchitecture {}

pub trait RollbackWritingBoundary {}
impl RollbackWritingBoundary for StructuralEditorArchitecture {}

pub trait CompilerArtifactBoundary {}
impl CompilerArtifactBoundary for StructuralEditorArchitecture {}

pub trait CompilerBufferBoundary {}
impl CompilerBufferBoundary for StructuralEditorArchitecture {}

pub trait CompilerTextBoundary {}
impl CompilerTextBoundary for StructuralEditorArchitecture {}

pub trait CompilerEdgeBoundary {}
impl CompilerEdgeBoundary for StructuralEditorArchitecture {}

pub trait CompilerCargoBoundary {}
impl CompilerCargoBoundary for StructuralEditorArchitecture {}

pub trait CompilerVerifyBoundary {}
impl CompilerVerifyBoundary for StructuralEditorArchitecture {}

pub trait RendererDispatchBoundary {}
impl RendererDispatchBoundary for StructuralEditorArchitecture {}

pub trait RendererFallbackBoundary {}
impl RendererFallbackBoundary for StructuralEditorArchitecture {}

pub trait ExecutorCompilationBoundary {}
impl ExecutorCompilationBoundary for StructuralEditorArchitecture {}

pub trait ExecutorCommitBoundary {}
impl ExecutorCommitBoundary for StructuralEditorArchitecture {}

pub trait ExecutorRollbackBoundary {}
impl ExecutorRollbackBoundary for StructuralEditorArchitecture {}

pub trait ExecutorReceiptBoundary {}
impl ExecutorReceiptBoundary for StructuralEditorArchitecture {}

pub trait ApiInputBoundary {}
impl ApiInputBoundary for StructuralEditorArchitecture {}

pub trait ApiOutputBoundary {}
impl ApiOutputBoundary for StructuralEditorArchitecture {}

pub trait JsonSchemaBoundary {}
impl JsonSchemaBoundary for StructuralEditorArchitecture {}

pub trait ErrorContextBoundary {}
impl ErrorContextBoundary for StructuralEditorArchitecture {}

pub trait UnsupportedOperationBoundary {}
impl UnsupportedOperationBoundary for StructuralEditorArchitecture {}

pub trait CompilerProofInputBoundary {}
impl CompilerProofInputBoundary for StructuralEditorArchitecture {}

pub trait CompilerProofOutputBoundary {}
impl CompilerProofOutputBoundary for StructuralEditorArchitecture {}

pub trait AtomicWorkspaceBoundary {}
impl AtomicWorkspaceBoundary for StructuralEditorArchitecture {}

/// Complete compile-time witness for the public structural editor pipeline.
///
/// Requiring this single bound pulls every phase capability through the type
/// system: callers cannot claim to own the complete editor boundary unless the
/// witness also satisfies the operation, compiler, renderer, executor, and API
/// boundary traits below.
pub trait CompleteStructuralEditorBoundary:
    TypedOperationBoundary
    + LocatorResolutionBoundary
    + TextSelectionBoundary
    + NodeCreationBoundary
    + NodeDeletionBoundary
    + NodeReplacementBoundary
    + NodeMoveBoundary
    + RenameBoundary
    + AttributeBoundary
    + EdgeInsertionBoundary
    + EdgeRemovalBoundary
    + VerificationBoundary
    + CargoManifestBoundary
    + ReceiptBoundary
    + RollbackBoundary
    + BufferedMutationBoundary
    + AtomicCommitBoundary
    + FilesystemCommitBoundary
    + DiffRenderingBoundary
    + JsonInputBoundary
    + JsonOutputBoundary
    + BatchLabelBoundary
    + ErrorProjectionBoundary
    + AllOrNothingBoundary
    + SelectorSyntaxBoundary
    + AnchorSpanBoundary
    + RenderedTextBoundary
    + UseEdgeBoundary
    + ModuleEdgeBoundary
    + ImplEdgeBoundary
    + BoundEdgeBoundary
    + PathRewriteBoundary
    + ExternCrateBoundary
    + FilePresenceBoundary
    + TextPresenceBoundary
    + SymbolPresenceBoundary
    + CompilerProofBoundary
    + FacadePreservationBoundary
    + ManifestChangeBoundary
    + ReceiptArtifactBoundary
    + RollbackArtifactBoundary
    + OperationStreamBoundary
    + BatchCompilationBoundary
    + OperationDispatchBoundary
    + CommitterBoundary
    + BatchExecutionBoundary
    + EditorApiBoundary
    + CliBoundary
    + OperationValidationBoundary
    + OperationNormalizationBoundary
    + LocatorNormalizationBoundary
    + SelectorResolutionBoundary
    + AnchorResolutionBoundary
    + TextPatchPlanningBoundary
    + FileDeltaProjectionBoundary
    + EditResultProjectionBoundary
    + ReceiptWritingBoundary
    + RollbackWritingBoundary
    + CompilerArtifactBoundary
    + CompilerBufferBoundary
    + CompilerTextBoundary
    + CompilerEdgeBoundary
    + CompilerCargoBoundary
    + CompilerVerifyBoundary
    + RendererDispatchBoundary
    + RendererFallbackBoundary
    + ExecutorCompilationBoundary
    + ExecutorCommitBoundary
    + ExecutorRollbackBoundary
    + ExecutorReceiptBoundary
    + ApiInputBoundary
    + ApiOutputBoundary
    + JsonSchemaBoundary
    + ErrorContextBoundary
    + UnsupportedOperationBoundary
    + CompilerProofInputBoundary
    + CompilerProofOutputBoundary
    + AtomicWorkspaceBoundary
{
}

impl CompleteStructuralEditorBoundary for StructuralEditorArchitecture {}
