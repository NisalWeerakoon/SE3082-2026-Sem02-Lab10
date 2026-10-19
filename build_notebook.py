"""Create the Colab notebook directly from the reviewed CUDA sources."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cells = []


def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": source.splitlines(keepends=True)})


def text(source):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": source.splitlines(keepends=True)})


def write_file(filename):
    code(f"%%writefile {filename}\n" + (root / filename).read_text(encoding="utf-8"))


text("# SE3082 Parallel Computing — Lab 10\nStudent: IT24103467. Run on a T4 GPU. Outputs are from actual Colab runs, not sample data.")
code("!nvidia-smi\n!nvcc --version")
write_file("cuda_utils.cuh")

text("## Exercise 1: GPU timing\nThree trials for each Lab 9 program, plus 100-million-element transfer comparison. The clock stops only after `cudaDeviceSynchronize()`.")
write_file("ex1_timing.cu")
code("!nvcc -arch=sm_75 ex1_timing.cu -o ex1_timing && ./ex1_timing")
text("The one-block version should avoid the overhead of scheduling 512 tiny blocks, but both short timings may be noisy. The 100-million-element case does more work, though timing need not scale exactly tenfold due to memory throughput and launch overhead. Whole-sequence time includes host/device transfers and should be substantially larger than kernel-only time. Without synchronizing before stopping the clock, the apparent kernel time measures mostly launch overhead; the later blocking copy can still make verification pass.")

text("## Exercise 2: CHECK error handling\nThe intentional failures are expected and are run under shell guards so the notebook continues. `invalid` probes the launch configuration; `oom` probes device allocation.")
write_file("ex2_check.cu")
code("!nvcc -arch=sm_75 ex2_check.cu -o ex2_check && ./ex2_check\n!./ex2_check invalid || echo 'Expected invalid-configuration failure'\n!./ex2_check oom || echo 'Expected out-of-memory failure'")
text("`CHECK(cudaGetLastError())` catches invalid launch configurations; `CHECK(cudaMalloc(...))` catches allocation errors; `CHECK(cudaDeviceSynchronize())` catches execution failures. Without `CHECK`, an invalid launch might be noticed only through incorrect verification results.")

text("## Exercise 3: 800 × 600 image\nA 50 × 38 grid of 16 × 16 threads covers every pixel. Values above 127 saturate to 255 when doubled, so roughly half the pixels are capped.")
write_file("ex3_picture.cu")
code("!nvcc -arch=sm_75 ex3_picture.cu -o ex3_picture && ./ex3_picture")

text("## Exercise 4: 500 × 500 × 500 volume\nAn 8 × 8 × 8 block and 63 × 63 × 63 grid cover the volume with 3D bounds checks. This is element-wise multiplication, not linear-algebra matrix multiplication.")
write_file("ex4_volume.cu")
code("!nvcc -arch=sm_75 ex4_volume.cu -o ex4_volume && ./ex4_volume")
text("The 3D case does 125 million products versus 100 million in Lab 9's 2D case. Its time may differ for memory-access, scheduling, and cache reasons, not just the element count.")

notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "SE3082_Lab10_IT24103467.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
(root / "SE3082_Lab10_IT24103467.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
