import sys
import pybind11_stubgen
import triton_client
if __name__ == '__main__':
    print(triton_client)
    print(sys.executable)
    pybind11_stubgen.main(["triton_client", "-o", "stubs"])
