import ctypes
import sys
import pathlib
import numpy as np

lib_name = ""
if sys.platform.startswith("linux"):
    lib_name = "libedgegpt.so"
elif sys.platform.startswith("win32"):
    lib_name = "edgegpt.dll"
elif sys.platform.startswith("darwin"):
    lib_name = "libedgegpt.dylib"
else:
    raise ImportError("Unsupported operating system")

try:
    lib_path = "/home/olafurj/dev/edgebert/target/release/" + lib_name
    edge_lib = ctypes.CDLL(str(lib_path))
except OSError as e:
    print(f"Error: Could not load the library '{lib_name}'.")
    print(f"Please make sure '{lib_name}' is in the same directory as this script.")
    print(f"Details: {e}")
    sys.exit(1)

print(f"Successfully loaded library: {lib_name}")

class EdgeGptDevice(ctypes.c_int):
    Cpu = 0
    Gpu = 1

class CEmbeddingResult(ctypes.Structure):
    _fields_ = [
        ("embeddings_ptr", ctypes.POINTER(ctypes.c_float)),
        ("num_embeddings", ctypes.c_size_t),
        ("embedding_dim", ctypes.c_size_t),
        ("error_message", ctypes.c_char_p),
    ]

edge_lib.edgegpt_create.argtypes = [EdgeGptDevice]
edge_lib.edgegpt_create.restype = ctypes.c_void_p
edge_lib.edgegpt_destroy.argtypes = [ctypes.c_void_p]
edge_lib.edgegpt_destroy.restype = None
edge_lib.edgegpt_encode_batch.argtypes = [
    ctypes.c_void_p,                     # handle
    ctypes.POINTER(ctypes.c_char_p),     # sentences_ptr
    ctypes.c_int                         # num_sentences
]
edge_lib.edgegpt_encode_batch.restype = ctypes.POINTER(CEmbeddingResult)
edge_lib.edgegpt_free_embedding_result.argtypes = [ctypes.POINTER(CEmbeddingResult)]
edge_lib.edgegpt_free_embedding_result.restype = None

def main():
    print("\nCreating EdgeGPT engine for GPU...")
    engine_handle = edge_lib.edgegpt_create(EdgeGptDevice.Gpu)

    if not engine_handle:
        print("Failed to create EdgeGPT engine. Exiting.")
        return
    try:
        sentences = [
            "The cat sits on the mat",
            "A feline rests on a rug",
            "Dogs are playing in the park",
        ]

        c_sentences = (ctypes.c_char_p * len(sentences))()
        encoded_sentences = [s.encode('utf-8') for s in sentences]
        c_sentences[:] = encoded_sentences

        print(f"\nEncoding {len(sentences)} sentences...")
        
        result_ptr = edge_lib.edgegpt_encode_batch(
            engine_handle,
            c_sentences,
            len(sentences)
        )

        try:
            if not result_ptr:
                print("Encoding function returned a null pointer. An error likely occurred.")
                return

            result = result_ptr.contents
            
            if result.error_message:
                error = result.error_message.decode('utf-8')
                print(f"An error occurred during encoding: {error}")
            else:
                print("Encoding successful!")
                print(f"Number of embeddings: {result.num_embeddings}")
                print(f"Embedding dimension: {result.embedding_dim}")

                total_elements = result.num_embeddings * result.embedding_dim
                flat_embeddings = np.ctypeslib.as_array(result.embeddings_ptr, shape=(total_elements,))
                
                embeddings = flat_embeddings.reshape((result.num_embeddings, result.embedding_dim))

                print("\nEmbeddings (first 5 dimensions of each):")
                for i, emb in enumerate(embeddings):
                    print(f"  Sentence {i}: {emb[:5]}...")
                
                vec_a = embeddings[0]
                vec_b = embeddings[1]
                similarity = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
                print(f"\nCosine similarity between sentence 0 and 1: {similarity:.4f}")

        finally:
            print("\nFreeing encoding result memory...")
            edge_lib.edgegpt_free_embedding_result(result_ptr)

    finally:
        print("Destroying EdgeGPT engine...")
        edge_lib.edgegpt_destroy(engine_handle)
        print("Done.")


if __name__ == "__main__":
    main()