"""Write the Rust, C and C++ public APIs as Markdown pages.

    reference_native.py rust <rustdoc.json> <workspace-root> <out-dir>
    reference_native.py c <kjarni.h> <out-dir>
    reference_native.py cpp <kjarni.hpp> <out.md>

Rust reads rustdoc's JSON and takes signatures from the source, which is steadier
than rendering rustdoc's unstable type tree. C and C++ read the headers: kjarni.h
comes from cbindgen and kjarni.hpp keeps one declaration style, so a scanner that
tracks braces is enough. Rust and C get a page per module, since each runs to
thousands of lines. C and C++ fail on a public item with no doc comment, so the
reference cannot quietly fall behind the API.
"""

import json
import re
import sys
from pathlib import Path

# The Rust modules a user of the crate calls, in the order of the page.
RUST_MODULES = [
    "embedder", "classifier", "reranker", "indexer", "searcher", "chat",
    "transcriber", "summarizer", "translator", "seq2seq", "generation", "generator",
    "registry", "common",
]


def prose(text: str) -> str:
    """Doc text as Markdown under the page's own headings.

    A "# Safety" inside a doc comment would otherwise outrank the item it
    describes, and rustdoc's hidden "# use x;" example lines would show.
    """
    lines, fenced = [], False
    for line in text.strip().splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            lines.append(re.sub(r"```(ignore|no_run|rust,ignore|rust,no_run)\s*$", "```rust", line))
        elif fenced:
            if line.strip() != "#" and not line.lstrip().startswith("# "):
                lines.append(line)
        else:
            lines.append(re.sub(r"^#{1,6}\s+(.+)$", r"**\1**", line))
    return "\n".join(lines).strip()


def code(language: str, text: str) -> str:
    return f"```{language}\n{text.strip()}\n```\n\n"


# ─── Rust ──────────────────────────────────────────────────────────


def rust(json_path: str, root: str, out_path: str) -> None:
    data = json.loads(Path(json_path).read_text())
    index, root_dir = data["index"], Path(root)
    sources: dict[str, list[str]] = {}

    def item(item_id):
        return index.get(str(item_id))

    def source_of(entry) -> str:
        span = entry["span"]
        lines = sources.setdefault(
            span["filename"], (root_dir / span["filename"]).read_text().splitlines()
        )
        first, last = span["begin"][0] - 1, span["end"][0]
        taken = [
            l for l in lines[first:last]
            if not l.lstrip().startswith(("///", "//!", "#["))
        ]
        return "\n".join(taken)

    def signature(entry) -> str:
        text = source_of(entry)
        depth, end = 0, len(text)
        for i, c in enumerate(text):
            if c in "([<":
                depth += 1
            elif c in ")]>" and text[i - 1] != "-":
                depth -= 1
            elif c in "{;" and depth <= 0:
                end = i
                break
        return re.sub(r"\s+", " ", text[:end]).strip().replace("( ", "(").replace(", )", ")")

    def kind(entry) -> str:
        return next(iter(entry["inner"]))

    def public(entry) -> bool:
        return entry["visibility"] == "public"

    def inherent_methods(entry, inner):
        for impl_id in inner.get("impls", []):
            impl = item(impl_id)
            if not impl:
                continue
            details = impl["inner"]["impl"]
            if details.get("trait") or details.get("is_synthetic") or details.get("blanket_impl"):
                continue
            for method_id in details["items"]:
                method = item(method_id)
                if method and public(method) and kind(method) == "function":
                    yield method

    out = []
    crate = index[str(data["root"])]
    modules = {}
    for item_id in crate["inner"]["module"]["items"]:
        entry = item(item_id)
        if entry and kind(entry) == "module" and public(entry):
            modules[entry["name"]] = entry

    count = 0

    def write_item(entry, level):
        nonlocal count
        k = kind(entry)
        name = entry["name"]
        count += 1
        heading = "#" * level
        if k == "function":
            out.append(f"{heading} {name}\n\n" + code("rust", signature(entry)))
            out.append(prose(entry.get("docs") or "") + "\n\n")
        elif k == "struct":
            inner = entry["inner"]["struct"]
            out.append(f"{heading} {name}\n\n" + code("rust", signature(entry)))
            out.append(prose(entry.get("docs") or "") + "\n\n")
            plain = inner["kind"].get("plain") if isinstance(inner["kind"], dict) else None
            fields = [item(f) for f in (plain or {}).get("fields", [])]
            fields = [f for f in fields if f and public(f)]
            if fields:
                out.append("**Fields**\n\n")
                for f in fields:
                    doc = prose(f.get("docs") or "").replace("\n", " ")
                    out.append(f"- `{signature(f)}`" + (f": {doc}" if doc else "") + "\n")
                out.append("\n")
            for method in inherent_methods(entry, inner):
                write_method(method, level + 1)
        elif k == "enum":
            inner = entry["inner"]["enum"]
            out.append(f"{heading} {name}\n\n" + code("rust", signature(entry)))
            out.append(prose(entry.get("docs") or "") + "\n\n")
            variants = [item(v) for v in inner.get("variants", [])]
            if variants:
                out.append("**Variants**\n\n")
                for v in filter(None, variants):
                    doc = prose(v.get("docs") or "").replace("\n", " ")
                    out.append(f"- `{signature(v)}`" + (f": {doc}" if doc else "") + "\n")
                out.append("\n")
            for method in inherent_methods(entry, inner):
                write_method(method, level + 1)
        elif k == "trait":
            out.append(f"{heading} {name}\n\n" + code("rust", signature(entry)))
            out.append(prose(entry.get("docs") or "") + "\n\n")
        elif k in ("type_alias", "constant"):
            out.append(f"{heading} {name}\n\n" + code("rust", signature(entry)))
            out.append(prose(entry.get("docs") or "") + "\n\n")

    def write_method(method, level):
        out.append(f"{'#' * level} {method['name']}\n\n" + code("rust", signature(method)))
        out.append(prose(method.get("docs") or "") + "\n\n")

    def write_module(entry, path, seen, top=False):
        written = {}
        for item_id in entry["inner"]["module"]["items"]:
            child = item(item_id)
            if not child or not public(child):
                continue
            k = kind(child)
            if k == "use":
                target = item(child["inner"]["use"].get("id"))
                # Re-exports from this crate's private submodules are the module's API.
                if target and kind(target) in ("struct", "enum", "trait", "function", "type_alias", "constant"):
                    written.setdefault(target["id"], target)
            elif k in ("struct", "enum", "trait", "function", "type_alias", "constant"):
                written.setdefault(child["id"], child)
        fresh = [e for i, e in written.items() if i not in seen]
        if fresh:
            if not top:
                out.append(f"## {path}\n\n")
            out.append(prose(entry.get("docs") or "") + "\n\n")
            for child in sorted(fresh, key=lambda e: (kind(e) != "struct", e["name"])):
                seen.add(child["id"])
                write_item(child, 3)
        for item_id in entry["inner"]["module"]["items"]:
            child = item(item_id)
            if child and public(child) and kind(child) == "module":
                write_module(child, f"{path}::{child['name']}", seen)

    seen = set()
    directory = Path(out_path)
    directory.mkdir(parents=True, exist_ok=True)
    overview = [
        "# Overview\n\n",
        "The `kjarni` crate on crates.io, added with `cargo add kjarni`. Each capability is "
        "a module with a builder, and its main type is re-exported at the crate root. These "
        "pages cover those modules; [docs.rs](https://docs.rs/kjarni/latest/kjarni/) has the "
        "lower-level crates they are built on.\n\n",
        code("rust", """use kjarni::Embedder;

let embedder = Embedder::builder("minilm-l6-v2").build().await?;
let similarity = embedder.similarity("doctor", "physician").await?;"""),
        "**Modules:**\n\n",
    ]
    for name in RUST_MODULES:
        if name not in modules:
            continue
        out.clear()
        out.append(f"# kjarni::{name}\n\n")
        write_module(modules[name], name, seen, top=True)
        (directory / f"{name}.md").write_text("".join(out))
        summary = prose(modules[name].get("docs") or "").split("\n\n")[0].replace("\n", " ")
        overview.append(f"- [{name}]({name}.md)" + (f": {summary}" if summary else "") + "\n")
    (directory / "overview.md").write_text("".join(overview))
    print(f"rust: {count} items -> {directory}")


# ─── C ─────────────────────────────────────────────────────────────


def c_header(header: str, out_path: str) -> None:
    text = Path(header).read_text()
    pattern = re.compile(
        r"(?:/\*\*(?P<doc>(?:(?!\*/).)*?)\*/\s*)?"
        r"^(?P<decl>(?:typedef\s+(?:enum|struct)\b|typedef\s|struct\s+\w+\s*\{|enum\s+\w+\s*\{|[A-Za-z_][\w \*]*\bkjarni_\w+\()"
        r".*?;)\s*$",
        re.S | re.M,
    )

    def clean_doc(doc):
        if not doc:
            return ""
        return prose("\n".join(re.sub(r"^\s*\* ?", "", l) for l in doc.splitlines()))

    types, functions, undocumented = [], {}, []
    pos = text.find("#include <stdbool.h>")
    for match in pattern.finditer(text, pos):
        decl = match.group("decl").strip()
        doc = clean_doc(match.group("doc"))
        call = re.search(r"\b(kjarni_\w+)\(", decl)
        if call and not decl.startswith("typedef") and "{" not in decl.split("(")[0]:
            name = call.group(1)
            if not doc:
                undocumented.append(name)
            prefix = name.split("_")[1]
            # kjarni_index_info and kjarni_search_options_default belong with the
            # types they configure, and Zola reserves a page named "index".
            prefix = {"index": "indexer", "search": "searcher"}.get(prefix, prefix)
            one_line = re.sub(r"\s*\n\s*", " ", decl)
            functions.setdefault(prefix, []).append((name, one_line, doc))
        else:
            words = re.findall(r"\w+", decl.split("{")[-1] if "{" in decl else decl)
            types.append((words[-1], decl, doc))

    directory = Path(out_path)
    directory.mkdir(parents=True, exist_ok=True)
    # Prefixes with one or two functions (kjarni_last_error_message, ...) read
    # better together than as pages of their own.
    general = []
    for prefix in [p for p, items in functions.items() if len(items) < 3]:
        general.extend(functions.pop(prefix))
    if general:
        functions["general"] = general

    out = [
        "# Overview\n\n",
        "`kjarni.h` is the whole C API, and each [release archive]"
        "(https://github.com/olafurjohannsson/kjarni/releases) ships it with the shared "
        "library. Fallible functions return a `KjarniErrorCode`; "
        "`kjarni_last_error_message` describes the last failure on the calling thread. "
        "Anything the library allocates has a matching `_free` function.\n\n",
        "**Pages:**\n\n",
        "- [Types](types.md)\n",
    ]
    for prefix, items in functions.items():
        out.append(f"- [{prefix}]({prefix}.md): {len(items)} functions\n")
    (directory / "overview.md").write_text("".join(out))

    page = ["# Types\n\n"]
    for name, decl, doc in types:
        page.append(f"## {name}\n\n" + code("c", decl))
        if doc:
            page.append(doc + "\n\n")
    (directory / "types.md").write_text("".join(page))
    for prefix, items in functions.items():
        page = [f"# {prefix}\n\n"]
        for name, decl, doc in items:
            page.append(f"## {name}\n\n" + code("c", decl))
            if doc:
                page.append(doc + "\n\n")
        (directory / f"{prefix}.md").write_text("".join(page))
    total = sum(len(v) for v in functions.values())
    print(f"c: {len(types)} types, {total} functions -> {directory}")
    if undocumented:
        sys.exit(f"{len(undocumented)} C functions have no doc comment: {', '.join(undocumented)}")


# ─── C++ ───────────────────────────────────────────────────────────


def strip_strings(line: str) -> str:
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])'", "''", line)
    return line.split("//")[0]


def cpp_header(header: str, out_path: str) -> None:
    lines = Path(header).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("namespace kjarni {"))
    intro = []
    for line in lines[:start]:
        if not line.startswith("//"):
            break
        intro.append(line[2:].removeprefix(" "))
    i, doc = start + 1, []
    top = []  # (kind, name, signature-or-body, doc, members)
    undocumented = []

    def balance(text):
        cleaned = strip_strings(text)
        return cleaned.count("{") - cleaned.count("}")

    def block_end(first):
        depth, j = 0, first
        while True:
            depth += balance(lines[j])
            if depth <= 0 and ("{" in strip_strings("\n".join(lines[first:j + 1]))):
                return j
            if depth <= 0 and strip_strings(lines[j]).rstrip().endswith(";"):
                return j
            j += 1

    def statement(first):
        """A declaration from `first` up to its opening brace or semicolon."""
        j, text = first, []
        while True:
            # `= {}` is a default argument, not the start of a body.
            cleaned = strip_strings(lines[j]).replace("{}", "\0\0")
            brace, semi = cleaned.find("{"), cleaned.find(";")
            cuts = [c for c in (brace, semi) if c >= 0]
            if cuts:
                text.append(cleaned[: min(cuts)])
                joined = " ".join(text).replace("\0\0", "{}")
                if joined.rstrip().endswith("{}") and ")" not in joined.rsplit("{}", 1)[1]:
                    joined = joined.rstrip()[:-2]
                return re.sub(r"\s+", " ", joined).strip(), j
            text.append(cleaned)
            j += 1

    def parse_members(first, last):
        members, pending, public, j = [], [], False, first
        while j < last:
            raw = lines[j].strip()
            if raw in ("public:", "private:", "protected:"):
                public = raw == "public:"
                pending, j = [], j + 1
                continue
            if raw.startswith("///"):
                pending.append(raw[3:].removeprefix(" "))
                j += 1
                continue
            if not raw or raw.startswith("//") or raw.startswith("#"):
                j += 1
                continue
            end = block_end(j)
            if public and not raw.startswith(("using ", "friend ", "~")):
                sig, _ = statement(j)
                sig = re.sub(r"\[\[nodiscard\]\]\s*", "", sig)
                sig = re.sub(r"\s*:\s*handle_.*$", "", sig)
                if "= delete" not in sig and "= default" not in sig and "operator=" not in sig:
                    members.append((sig, list(pending)))
            pending, j = [], end + 1
        return members

    while i < len(lines):
        raw = lines[i]
        if raw.startswith("///"):
            doc.append(raw[3:].removeprefix(" "))
            i += 1
            continue
        if raw.startswith("namespace detail"):
            depth, j = 0, i
            while True:
                depth += balance(lines[j])
                if depth == 0 and j > i or (depth == 0 and "{" in lines[j]):
                    break
                j += 1
            i, doc = j + 1, []
            continue
        match = re.match(r"(class|struct)\s+(\w+)\s*(?:[:{]|$)", raw)
        enum = re.match(r"enum class\s+(\w+)", raw)
        func = raw.startswith(("[[nodiscard]] inline", "inline ", "[[nodiscard]] constexpr"))
        alias = re.match(r"using\s+(\w+)\s*=", raw)
        if raw.startswith("template"):
            i += 1
            continue
        if match:
            end = block_end(i)
            name = match.group(2)
            if match.group(1) == "struct":
                body = [l[4:] for l in lines[i + 1:end] if l.strip()]
                top.append(("struct", name, "\n".join(body), doc, []))
            else:
                members = parse_members(i + 1, end)
                top.append(("class", name, f"class {name}", doc, members))
            i, doc = end + 1, []
            continue
        if enum:
            top.append(("enum", enum.group(1), raw.strip(), doc, []))
            i, doc = block_end(i) + 1, []
            continue
        if func:
            sig, _ = statement(i)
            sig = re.sub(r"\[\[nodiscard\]\]\s*|\binline\s+", "", sig)
            top.append(("function", re.search(r"(\w+)\(", sig).group(1), sig, doc, []))
            i, doc = block_end(i) + 1, []
            continue
        if alias:
            top.append(("alias", alias.group(1), raw.strip(), doc, []))
            i, doc = i + 1, []
            continue
        if raw.strip():
            doc = []
        i += 1

    # The header's opening comment is prose around one indented example.
    out = ["# C++ API\n\n"]
    example = [l[2:] for l in intro if l.startswith("  ")]
    text = [l for l in intro if not l.startswith("  ")]
    first_example = next(i for i, l in enumerate(intro) if l.startswith("  "))
    before = sum(1 for l in intro[:first_example] if not l.startswith("  "))
    out.append("\n".join(text[:before]).strip() + "\n\n")
    out.append(code("cpp", "\n".join(example)))
    out.append("\n".join(text[before:]).strip() + "\n\n")
    out.append("## Types\n\n")
    for kind, name, sig, comment, members in top:
        if kind in ("function",):
            continue
        out.append(f"### {name}\n\n")
        if comment:
            out.append(prose("\n".join(comment)) + "\n\n")
        if kind == "class":
            for member_sig, member_doc in members:
                method = re.search(r"(\w+)\(", member_sig)
                title = method.group(1) if method else member_sig
                out.append(f"#### {title}\n\n" + code("cpp", member_sig))
                if member_doc:
                    out.append(prose("\n".join(member_doc)) + "\n\n")
                elif title != name:
                    undocumented.append(f"{name}::{title}")
        else:
            out.append(code("cpp", sig))
    functions = [t for t in top if t[0] == "function"]
    if functions:
        out.append("## Functions\n\n")
        for _, name, sig, comment, _ in functions:
            out.append(f"### {name}\n\n" + code("cpp", sig))
            if comment:
                out.append(prose("\n".join(comment)) + "\n\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(out))
    print(f"cpp: {len(top)} items -> {out_path}")
    if undocumented:
        sys.exit(f"{len(undocumented)} C++ members have no doc comment: {', '.join(undocumented)}")


if __name__ == "__main__":
    {"rust": rust, "c": c_header, "cpp": cpp_header}[sys.argv[1]](*sys.argv[2:])
