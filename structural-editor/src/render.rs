#![allow(dead_code)]
use crate::op::NodeKind;

/// Typed fields required to render a structural node.
pub struct RenderNodeInput<'a> {
    pub visibility: Option<&'a str>,
    pub name: &'a str,
    pub body: &'a str,
}

impl<'a> RenderNodeInput<'a> {
    pub fn new(visibility: Option<&'a str>, name: Option<&'a str>, body: Option<&'a str>) -> Self {
        Self {
            visibility,
            name: name.unwrap_or("unnamed"),
            body: body.unwrap_or(""),
        }
    }
}

/// Renders one structural node category from typed rendering input.
pub trait NodeRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FileRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct UseDeclRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct ModDeclRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct FunctionRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct StructRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct EnumRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct TraitRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct ImplBlockRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct TraitImplRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct TypeAliasRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct ConstRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct StaticRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct StructFieldRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct EnumVariantRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct WhereClauseRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct GenericParamRenderer;
#[derive(Debug, Default, Clone, Copy)]
pub struct RawBodyRenderer;

impl NodeRenderer for FileRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_body(input)
    }
}

impl NodeRenderer for UseDeclRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_use(input.name)
    }
}

impl NodeRenderer for ModDeclRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_mod(input.name, false, None)
    }
}

impl NodeRenderer for FunctionRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_function(input.visibility, input.name, input.body)
    }
}

impl NodeRenderer for StructRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_keyword_item(input, "struct")
    }
}

impl NodeRenderer for EnumRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_keyword_item(input, "enum")
    }
}

impl NodeRenderer for TraitRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_keyword_item(input, "trait")
    }
}

impl NodeRenderer for ImplBlockRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_impl_like(input)
    }
}

impl NodeRenderer for TraitImplRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_impl_like(input)
    }
}

impl NodeRenderer for TypeAliasRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_type_alias(input.visibility, input.name, input.body)
    }
}

impl NodeRenderer for ConstRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_const_or_static(input, "const")
    }
}

impl NodeRenderer for StaticRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_const_or_static(input, "static")
    }
}

impl NodeRenderer for StructFieldRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_field(input.visibility, input.name, input.body)
    }
}

impl NodeRenderer for EnumVariantRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_variant(input.name, Some(input.body).filter(|s| !s.is_empty()))
    }
}

impl NodeRenderer for WhereClauseRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_where(input.name, input.body)
    }
}

impl NodeRenderer for GenericParamRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_generic_param(input.name)
    }
}

impl NodeRenderer for RawBodyRenderer {
    fn render(&self, input: &RenderNodeInput<'_>) -> String {
        render_body(input)
    }
}

fn render_body(input: &RenderNodeInput<'_>) -> String {
    input.body.to_string()
}

fn render_keyword_item(input: &RenderNodeInput<'_>, keyword: &str) -> String {
    render_item(input.visibility, keyword, input.name, input.body)
}

fn render_impl_like(input: &RenderNodeInput<'_>) -> String {
    render_impl(input.name, input.body)
}

fn render_const_or_static(input: &RenderNodeInput<'_>, keyword: &str) -> String {
    render_const(
        keyword,
        input.visibility,
        input.name,
        "/* type */",
        input.body,
    )
}

fn render_vis_prefixed_decl(visibility: Option<&str>, decl: String) -> String {
    let vis = vis_prefix(visibility);
    format!("{vis}{decl}")
}

fn render_line(content: String) -> String {
    format!("{content}\n")
}

fn render_semicolon_decl(content: String) -> String {
    render_line(format!("{content};"))
}

fn render_braced_block(header: &str, body: &str) -> String {
    render_line(format!("{header} {{\n{body}\n}}"))
}

/// Resolve a node kind to the renderer that owns its formatting rules.
pub fn renderer_for(kind: &NodeKind) -> &'static dyn NodeRenderer {
    match kind {
        NodeKind::File => &FileRenderer,
        NodeKind::UseDecl => &UseDeclRenderer,
        NodeKind::ModDecl => &ModDeclRenderer,
        NodeKind::Function | NodeKind::ImplItem | NodeKind::TraitItem => &FunctionRenderer,
        NodeKind::Struct => &StructRenderer,
        NodeKind::Enum => &EnumRenderer,
        NodeKind::Trait => &TraitRenderer,
        NodeKind::ImplBlock => &ImplBlockRenderer,
        NodeKind::TraitImpl => &TraitImplRenderer,
        NodeKind::TypeAlias => &TypeAliasRenderer,
        NodeKind::Const => &ConstRenderer,
        NodeKind::Static => &StaticRenderer,
        NodeKind::StructField => &StructFieldRenderer,
        NodeKind::EnumVariant => &EnumVariantRenderer,
        NodeKind::WhereClause => &WhereClauseRenderer,
        NodeKind::GenericParam | NodeKind::Lifetime => &GenericParamRenderer,
        _ => &RawBodyRenderer,
    }
}

/// Render a complete Rust module block.
pub fn render_module(name: &str, body: &str) -> String {
    render_braced_block(&format!("mod {name}"), body)
}

/// Render a function or method item.
pub fn render_function(visibility: Option<&str>, sig: &str, body: &str) -> String {
    render_vis_prefixed_decl(visibility, render_braced_block(&format!("fn {sig}"), body))
}

/// Render a struct, enum, or trait item.
pub fn render_item(visibility: Option<&str>, keyword: &str, sig: &str, body: &str) -> String {
    render_vis_prefixed_decl(
        visibility,
        render_braced_block(&format!("{keyword} {sig}"), body),
    )
}

/// Render an inherent impl block.
pub fn render_impl(sig: &str, body: &str) -> String {
    render_braced_block(&format!("impl {sig}"), body)
}

/// Render a trait impl block (`impl Trait for Type`).
pub fn render_trait_impl(sig: &str, body: &str) -> String {
    render_braced_block(&format!("impl {sig}"), body)
}

/// Render a `#[test]` fn.
pub fn render_test(name: &str, body: &str) -> String {
    render_line(format!("#[test]\nfn {name}() {{\n    {body}\n}}"))
}

/// Render a struct field: `    pub name: Type,`
pub fn render_field(visibility: Option<&str>, name: &str, ty: &str) -> String {
    let decl = render_vis_prefixed_decl(visibility, format!("{name}: {ty},\n"));
    format!("    {decl}")
}

/// Render an enum variant.
pub fn render_variant(name: &str, body: Option<&str>) -> String {
    match body {
        Some(b) if !b.is_empty() => format!("    {name}({b}),\n"),
        _ => format!("    {name},\n"),
    }
}

/// Render a `use` declaration.
pub fn render_use(path: &str) -> String {
    render_semicolon_decl(format!("use {path}"))
}

/// Render a `mod` declaration.
pub fn render_mod(name: &str, inline: bool, body: Option<&str>) -> String {
    if inline {
        render_braced_block(&format!("mod {name}"), body.unwrap_or(""))
    } else {
        render_semicolon_decl(format!("mod {name}"))
    }
}

/// Render a type alias.
pub fn render_type_alias(visibility: Option<&str>, name: &str, ty: &str) -> String {
    render_vis_prefixed_decl(visibility, render_semicolon_decl(format!("type {name} = {ty}")))
}

/// Render a const or static.
pub fn render_const(
    keyword: &str,
    visibility: Option<&str>,
    name: &str,
    ty: &str,
    value: &str,
) -> String {
    render_vis_prefixed_decl(
        visibility,
        render_semicolon_decl(format!("{keyword} {name}: {ty} = {value}")),
    )
}

/// Render a where clause line.
pub fn render_where(param: &str, bound: &str) -> String {
    format!("    {param}: {bound},\n")
}

/// Render a generic parameter.
pub fn render_generic_param(param: &str) -> String {
    param.to_string()
}

#[allow(dead_code)]
/// Render a `[[bin]]`/`[[test]]`/`[[example]]` Cargo target stanza.
pub fn render_cargo_target(kind: &str, name: &str, path: &str) -> String {
    render_line(format!("[[{kind}]]\nname = \"{name}\"\npath = \"{path}\""))
}

/// Dispatch to the correct renderer by NodeKind given raw text fields.
/// Primarily useful when the caller doesn't want to pick a renderer manually.
pub fn render_node(
    kind: &NodeKind,
    visibility: Option<&str>,
    name: Option<&str>,
    body: Option<&str>,
) -> String {
    let input = RenderNodeInput::new(visibility, name, body);
    renderer_for(kind).render(&input)
}

fn vis_prefix(visibility: Option<&str>) -> String {
    match visibility {
        Some(v) if !v.is_empty() => format!("{v} "),
        _ => String::new(),
    }
}
