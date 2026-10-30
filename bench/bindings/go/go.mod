module kjarni-binding-overhead

go 1.24

require github.com/olafurjohannsson/kjarni-go v0.0.0

require github.com/ebitengine/purego v0.8.2 // indirect

replace github.com/olafurjohannsson/kjarni-go => ../../../crates/kjarni-ffi/bindings/go
