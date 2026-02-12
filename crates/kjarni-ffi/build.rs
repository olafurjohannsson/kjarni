fn main() {
    // Only generate header when c-bindings feature is enabled
    #[cfg(feature = "c-bindings")]
    {
        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let output_dir = std::path::Path::new(&crate_dir).join("include");
        
        // Create include directory if it doesn't exist
        std::fs::create_dir_all(&output_dir).expect("Failed to create include directory");
        
        let config = cbindgen::Config::from_file("cbindgen.toml")
            .unwrap_or_default();
        
        cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .with_config(config)
            .generate()
            .expect("Unable to generate C bindings")
            .write_to_file(output_dir.join("kjarni.h"));
        
        println!("cargo:rerun-if-changed=src/");
        println!("cargo:rerun-if-changed=cbindgen.toml");
    }
}