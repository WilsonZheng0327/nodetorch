"""Format the frontend-supplied node catalog into compact prompt text.

The catalog is the single source of truth from the frontend registry (see
src/domain/catalog.ts). The backend treats it purely as data — it never owns or
duplicates the node schema. Each entry:

    {
      "type": "ml.layers.conv2d",
      "displayName": "Conv2d",
      "category": ["ML", "Layers", "Convolution"],
      "description": "...",
      "properties": [{"id","name","kind","default", ...}],
      "ports": [{"id","direction","dataType","allowMultiple","optional"}],
      "modes": ["shape","forward","train"]
    }
"""
from __future__ import annotations


def _format_property(prop: dict) -> str:
    pid = prop.get("id", "?")
    kind = prop.get("kind", "")
    detail = ""
    if kind == "number":
        bits = []
        if prop.get("integer"):
            bits.append("int")
        if prop.get("min") is not None or prop.get("max") is not None:
            bits.append(f"{prop.get('min', '')}..{prop.get('max', '')}")
        detail = " ".join(bits)
    elif kind == "select":
        opts = prop.get("options") or []
        detail = "|".join(str(o.get("value", o)) for o in opts)
    elif kind in ("string", "boolean"):
        detail = kind
    default = prop.get("default")
    default_str = f" =default {default}" if default is not None else ""
    inner = f" ({detail})" if detail else ""
    return f"{pid}{inner}{default_str}"


def _format_ports(ports: list[dict]) -> str:
    ins = [p for p in ports if p.get("direction") == "input"]
    outs = [p for p in ports if p.get("direction") == "output"]

    def fmt(p: dict) -> str:
        tag = p.get("dataType", "")
        flags = []
        if p.get("optional"):
            flags.append("opt")
        if p.get("allowMultiple"):
            flags.append("multi")
        flag_str = f",{'/'.join(flags)}" if flags else ""
        return f"{p.get('id', '?')}:{tag}{flag_str}"

    parts = []
    if ins:
        parts.append("in[" + ", ".join(fmt(p) for p in ins) + "]")
    if outs:
        parts.append("out[" + ", ".join(fmt(p) for p in outs) + "]")
    return " ".join(parts)


def format_catalog(catalog: list[dict]) -> str:
    """Render the catalog as a compact, category-grouped reference block."""
    if not catalog:
        return "(node catalog unavailable)"

    by_category: dict[str, list[dict]] = {}
    for node in catalog:
        cat = " / ".join(node.get("category", [])) or "Other"
        by_category.setdefault(cat, []).append(node)

    lines: list[str] = []
    for cat in sorted(by_category):
        lines.append(f"## {cat}")
        for node in sorted(by_category[cat], key=lambda n: n.get("displayName", "")):
            header = f"- {node.get('type')} ({node.get('displayName')})"
            desc = node.get("description")
            if desc:
                header += f" — {desc}"
            lines.append(header)
            ports = node.get("ports") or []
            if ports:
                lines.append(f"    ports: {_format_ports(ports)}")
            props = node.get("properties") or []
            if props:
                lines.append(
                    "    props: " + ", ".join(_format_property(p) for p in props)
                )
    return "\n".join(lines)
