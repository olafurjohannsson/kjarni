// Writes the Go binding's public API as one Markdown page.
//
//	go run scripts/reference_go.go <package-dir> <out.md>
//
// Fails if an exported type, function or method has no doc comment, so the
// reference cannot quietly fall behind the API.
package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var fset = token.NewFileSet()
var out bytes.Buffer
var undocumented []string

func source(node any) string {
	var b bytes.Buffer
	if err := printer.Fprint(&b, fset, node); err != nil {
		panic(err)
	}
	return b.String()
}

// The declaration with its body removed: only the signature belongs on the page.
func signature(f *ast.FuncDecl) string {
	sig := *f
	sig.Body = nil
	sig.Doc = nil
	return source(&sig)
}

func emit(heading, code, comment, where string) {
	if strings.TrimSpace(comment) == "" {
		undocumented = append(undocumented, where)
	}
	fmt.Fprintf(&out, "%s %s\n\n```go\n%s\n```\n\n", heading, where, code)
	if text := markdown(comment); text != "" {
		out.WriteString(text + "\n\n")
	}
}

// Go doc comments indent code by a tab and mark headings with "# ".
func markdown(comment string) string {
	var pkg doc.Package
	parsed := pkg.Parser().Parse(comment)
	return strings.TrimSpace(string(pkg.Printer().Markdown(parsed)))
}

func main() {
	dir, outPath := os.Args[1], os.Args[2]
	pkgs, err := parser.ParseDir(fset, dir, func(fi os.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ParseComments)
	if err != nil {
		panic(err)
	}
	pkg := doc.New(pkgs["kjarni"], "github.com/olafurjohannsson/kjarni-go", 0)

	out.WriteString("# kjarni-go\n\n")
	// Package docs carry their own "# Heading"s, which sit under the page's sections.
	demote := regexp.MustCompile(`(?m)^#\s`)
	out.WriteString(demote.ReplaceAllString(markdown(pkg.Doc), "### ") + "\n\n")

	if len(pkg.Consts)+len(pkg.Vars) > 0 {
		out.WriteString("## Constants and variables\n\n")
		for _, v := range append(pkg.Consts, pkg.Vars...) {
			decl := *v.Decl
			decl.Doc = nil
			fmt.Fprintf(&out, "```go\n%s\n```\n\n", source(&decl))
			if text := markdown(v.Doc); text != "" {
				out.WriteString(text + "\n\n")
			}
		}
	}

	if len(pkg.Funcs) > 0 {
		out.WriteString("## Functions\n\n")
		for _, f := range pkg.Funcs {
			emit("###", signature(f.Decl), f.Doc, f.Name)
		}
	}

	out.WriteString("## Types\n\n")
	for _, t := range pkg.Types {
		decl := *t.Decl
		decl.Doc = nil
		emit("###", source(&decl), t.Doc, t.Name)
		for _, v := range t.Consts {
			d := *v.Decl
			d.Doc = nil
			fmt.Fprintf(&out, "```go\n%s\n```\n\n", source(&d))
		}
		for _, f := range t.Funcs {
			emit("####", signature(f.Decl), f.Doc, f.Name)
		}
		for _, m := range t.Methods {
			emit("####", signature(m.Decl), m.Doc, t.Name+"."+m.Name)
		}
	}

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		panic(err)
	}
	if err := os.WriteFile(outPath, out.Bytes(), 0o644); err != nil {
		panic(err)
	}
	if len(undocumented) > 0 {
		fmt.Fprintf(os.Stderr, "undocumented Go API: %s\n", strings.Join(undocumented, ", "))
		os.Exit(1)
	}
	fmt.Printf("go: %d types, %d functions -> %s\n", len(pkg.Types), len(pkg.Funcs), outPath)
}
