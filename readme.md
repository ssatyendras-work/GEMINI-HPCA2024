# Gemini Update  

We have iterated on Gemini, expanding it into a larger template. In the latest version, Gemini supports different process technologies, DRAM, D2D Interfaces, and packaging technologies. Currently, the latest Gemini supports:  

- **Two process technologies**: 7nm and 12nm  
- **Three D2D Interfaces**: XSR, USR, and UCIe  
- **Three types of DDR**: LPDDR5, GDDR6X, and HBM  
- **Three packaging technologies**: Organic Substrate (OS), Redistribution Layer(RDL), and Silicon Interposer(SI)  

For more details, please refer to the code.  

# The chess fork (`chess/main`)

This branch is upstream master plus what [chess](https://gitlab.com/ssatyendras/projects/technology-driven/chess) needs to drive Gemini from its own models; the input line and the result line stay upstream's.

- **Technology from a file (T3).** The first stdin token may be `@<file>` instead of `7` or `12`: the file holds every technology number `main.cpp` used to choose by branch (`include/technology.h`, `src/technology.cpp`; `stschedule --dump-technology <tech> <package> <IO> <DDR>` prints the built-in set in the file's form). The node key for `cost.cpp` is the file's `node` line.
- **Heterogeneous packages (H3).** An optional 21st token `@<chiplet map>` gives every chiplet of the `xcut x ycut` grid a chiplet type: core micro-architecture, MAC count, L2 size and its own technology file (`include/chiplets.h`, `src/chiplets.cpp`). Each type owns a core and a mapper; a layer spread over several types takes the slowest type's time and the sum of its cores' energies; die area and cost are summed per type; the package runs at the slowest type's clock and its NoC hop energy is the core-count-weighted mean. Without the token the package is upstream's and every number is identical. The map:

  ```
  types 2
  type 0 mm 0 mac 512 ul3_kb 1024 tech technology.tech
  type 1 mm 0 mac 1024 ul3_kb 512 tech technology.7nm.tech
  map 0 1      # one line per chiplet row (y), xcut entries
  map 1 0
  ```

  The run prints `CHIPLETS:` and one `CHIPLET_TYPE i:` line per type (node, clock, mm, MACs, L2, chiplet and core counts, core and die areas).
- **Fixed seed.** `GEMINI_SEED=<n>` seeds the search instead of the clock.

# How to Run the Project  

First, clone the entire project from GitHub:  

```bash
git clone https://github.com/SET-Scheduling-Project/GEMINI-HPCA2024.git
```
Then, navigate to the project directory:
```bash
cd GEMINI-HPCA2024
make
```
After building, the executable target will be generated at ./build/stschedule.

Next, install the required dependencies:
```bash
pip install -r requirements.txt
```
Now, you can run the DSE script, such as:
```bash
./72tops_dse13.sh
```
Wait for the execution to complete, and you will obtain the architecture exploration results. You can modify the DSE files to perform your own design space exploration experiments.
# Clarification
Many essential foundational components of Gemini are shared with the SET framework from the same project. We are currently updating the architecture and documentation of the SET framework. For urgent inquiries about Gemini, one can refer to the comments in SET (updated at 2025.2.1). Additionally, we plan to migrate and update these annotations from SET to Gemini and include comments related to code unique to Gemini.

For the calculation of the packaging cost in this article, we referred to the article "Chiplet Actuary: A Quantitative Cost Model and Multi-Chiplet Architecture Exploration" published in 2022 DAC and had relevant discussions with the author. We would like to thank the author of the article.

# Citations ###
```
@inproceedings{10.1145/3579371.3589048,
author = {Cai, Jingwei and Wei, Yuchen and Wu, Zuotong and Peng, Sen and Ma, Kaisheng},
title = {Inter-layer Scheduling Space Definition and Exploration for Tiled Accelerators},
year = {2023},
isbn = {9798400700958},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3579371.3589048},
doi = {10.1145/3579371.3589048},
booktitle = {Proceedings of the 50th Annual International Symposium on Computer Architecture},
articleno = {13},
numpages = {17},
keywords = {tiled accelerators, neural networks, inter-layer scheduling, scheduling},
location = {Orlando, FL, USA},
series = {ISCA '23}
}

```
```
@inproceedings{cai2024gemini,
  title={Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators},
  author={Cai, Jingwei and Wu, Zuotong and Peng, Sen and Wei, Yuchen and Tan, Zhanhong and Shi, Guiming and Gao, Mingyu and Ma, Kaisheng},
  booktitle={2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
  pages={156--171},
  year={2024},
  organization={IEEE}
}
```
