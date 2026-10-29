"""Python API reference as Markdown, from the source tree, without importing it.

    python reference_python.py <bindings/python dir> <output.md>

Importing kjarni loads the native library, which a docs build does not have, so
the package is read statically with griffe. Only what `kjarni.__all__` exports
is documented; the ctypes structures in the submodules are an implementation
detail. Exits non-zero, listing them, if any exported class, function or public
method has no docstring.
"""

import ast
import re
import sys
from pathlib import Path

import griffe
import griffe2md

src, out = Path(sys.argv[1]), Path(sys.argv[2])

init = ast.parse((src / "kjarni" / "__init__.py").read_text())
exported = next(
    ast.literal_eval(node.value)
    for node in init.body
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets)
)

pkg = griffe.load("kjarni", search_paths=[str(src)], allow_inspection=False)
missing = []
for name in exported:
    obj = pkg.members[name]
    target = obj.final_target if obj.is_alias else obj
    if not (target.is_class or target.is_function):
        continue
    if not target.docstring:
        missing.append(f"kjarni.{name}")
    if target.is_class:
        for member_name, member in target.members.items():
            if member_name.startswith("_") or member.is_alias:
                continue
            if member.is_function and not member.docstring:
                missing.append(f"kjarni.{name}.{member_name}")

if missing:
    print("Python: missing docstrings:", file=sys.stderr)
    for m in missing:
        print(f"  {m}", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(src))
markdown = griffe2md.render_package_docs("kjarni")

# Keep the package overview and the sections for exported names; drop the
# submodule sections that follow them.
keep, kept = True, []
for line in markdown.splitlines():
    heading = re.match(r"^(#{2,4}) `kjarni\.([A-Za-z_]+)", line)
    if heading and len(heading.group(1)) == 3:
        keep = heading.group(2) in exported
    elif heading and len(heading.group(1)) == 2:
        keep = True
    if keep:
        kept.append(line)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(kept) + "\n")
print(f"python: {len(exported)} exports -> {out}")
