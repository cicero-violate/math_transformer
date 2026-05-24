#![allow(dead_code)]
use crate::op::NodeKind;

/// Render a complete Rust module block.
pub fn render_module(name: &str, body: &str) -> String {
    format!("mod {name} {{\n{body}\n}}\n")
}

/// Render a function or method item.
pub fn render_function(visibility: Option<&str>, sig: &str, body: &str) -> String {
    let vis = vis_prefix(visibility);
    format!("{vis}fn {sig} {{\n{body}\n}}\n")
}

/// Render a struct, enum, or trait item.
pub fn render_item(visibility: Option<&str>, keyword: &str, sig: &str, body: &str) -> String {
    let vis = vis_prefix(visibility);
    format!("{vis}{keyword} {sig} {{\n{body}\n}}\n")
}

/// Render an inherent impl block.
pub fn render_impl(sig: &str, body: &str) -> String {
    format!("impl {sig} {{\n{body}\n}}\n")
}

/// Render a trait impl block (`impl Trait for Type`).
pub fn render_trait_impl(sig: &str, body: &str) -> String {
    format!("impl {sig} {{\n{body}\n}}\n")
}

/// Render a `#[test]` fn.
pub fn render_test(name: &str, body: &str) -> String {
    format!("#[test]\nfn {name}() {{\n    {body}\n}}\n")
}

/// Render a struct field: `    pub name: Type,`
pub fn render_field(visibility: Option<&str>, name: &str, ty: &str) -> String {
    let vis = vis_prefix(visibility);
    format!("    {vis}{name}: {ty},\n")
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
    format!("use {path};\n")
}

/// Render a `mod` declaration.
pub fn render_mod(name: &str, inline: bool, body: Option<&str>) -> String {
    if inline {
        format!("mod {name} {{\n{}\n}}\n", body.unwrap_or(""))
    } else {
        format!("mod {name};\n")
    }
}

/// Render a type alias.
pub fn render_type_alias(visibility: Option<&str>, name: &str, ty: &str) -> String {
    let vis = vis_prefix(visibility);
    format!("{vis}type {name} = {ty};\n")
}

/// Render a const or static.
pub fn render_const(
    keyword: &str,
    visibility: Option<&str>,
    name: &str,
    ty: &str,
    value: &str,
) -> String {
    let vis = vis_prefix(visibility);
    format!("{vis}{keyword} {name}: {ty} = {value};\n")
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
    format!("[[{kind}]]\nname = \"{name}\"\npath = \"{path}\"\n")
}

/// Dispatch to the correct renderer by NodeKind given raw text fields.
/// Primarily useful when the caller doesn't want to pick a renderer manually.
pub fn render_node(
    kind: &NodeKind,
    visibility: Option<&str>,
    name: Option<&str>,
    body: Option<&str>,
) -> String {
    let name = name.unwrap_or("unnamed");
    let body = body.unwrap_or("");
    match kind {
        NodeKind::File => body.to_string(),
        NodeKind::UseDecl => render_use(name),
        NodeKind::ModDecl => render_mod(name, false, None),
        NodeKind::Function | NodeKind::ImplItem | NodeKind::TraitItem => {
            render_function(visibility, name, body)
        }
        NodeKind::Struct => render_item(visibility, "struct", name, body),
        NodeKind::Enum => render_item(visibility, "enum", name, body),
        NodeKind::Trait => render_item(visibility, "trait", name, body),
        NodeKind::ImplBlock => render_impl(name, body),
        NodeKind::TraitImpl => render_trait_impl(name, body),
        NodeKind::TypeAlias => render_type_alias(visibility, name, body),
        NodeKind::Const => render_const("const", visibility, name, "/* type */", body),
        NodeKind::Static => render_const("static", visibility, name, "/* type */", body),
        NodeKind::StructField => render_field(visibility, name, body),
        NodeKind::EnumVariant => render_variant(name, Some(body).filter(|s| !s.is_empty())),
        NodeKind::WhereClause => render_where(name, body),
        NodeKind::GenericParam | NodeKind::Lifetime => render_generic_param(name),
        _ => body.to_string(),
    }
}

fn vis_prefix(visibility: Option<&str>) -> String {
    match visibility {
        Some(v) if !v.is_empty() => format!("{v} "),
        _ => String::new(),
    }
}
