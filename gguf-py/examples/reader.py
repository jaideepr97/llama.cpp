#!/usr/bin/env python3
import sys
from pathlib import Path
from gguf.gguf_reader import GGUFReader
from safetensors.torch import save_file
import os


sys.path.insert(0, str(Path(__file__).parent.parent))


def read_gguf_file(gguf_file_path, dest_dir):
    """
    Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.

    Parameters:
    - gguf_file_path: Path to the GGUF file.
    """

    reader = GGUFReader(gguf_file_path)

    # List all key-value pairs in a columnized format
    print("Key-Value Pairs:")
    max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        print(f"{key:{max_key_length}} : {value}")
    print("----")

    # List all tensors
    print("Tensors:")
    tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
    print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization"))
    print("-" * 80)
    for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        print(tensor_info_format.format(tensor.name, shape_str, size_str, quantization_str))
    
    # print("weights:")
    # for tensor in reader.tensors:
    #     print(tensor.data)

    save_file(
            reader.weights,
            os.path.join(dest_dir, "model.safetensors"),
            metadata={"format": "pt"},
        )

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: reader.py <path_to_gguf_file>")
        sys.exit(1)
    gguf_file_path = sys.argv[1]
    dest_dir = sys.argv[2]
    read_gguf_file(gguf_file_path, dest_dir)


