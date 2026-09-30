# ExploreChiplets  

This branch contains the **fine-grained simulation code for Gemini**, enhanced with **Timeloop-based predictors**.  

---

## 🔧 Modifications  

1. Python scripts for **Timeloop data generation**  
2. Python scripts for **predictor training**  
3. C++ code for **predictor inference**  

---

## ⚙️ Environment Setup  

### 1. Outside Docker (Python with GPU)

**Requirements:**  
- XGBoost
- sklearn
- Pytorch


### 2. Inside Docker  

**Requirements:**  
- XGBoost (API)  
- Timeloop (API)  

#### Install Docker and Timeloop  
```bash
# in bash
git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git
```

Create a `docker-compose.yaml` in your working folder, then modify as needed:  

```yaml
services:
   infrastructure:
      image: timeloopaccelergy/accelergy-timeloop-infrastructure:latest-amd64
      container_name: chiplet4nn
      user: "${MY_UID}:${MY_GID}"
      volumes:
         - [your_folder_path_in_bash]:/home/workspace:Z
      working_dir: /home/workspace
      environment:
         - HOME=/home/workspace
         - XGB_LIB=/home/workspace/.local/xgboost/lib
         - LD_LIBRARY_PATH=/home/workspace/.local/xgboost/lib:${LD_LIBRARY_PATH:-}
      stdin_open: true
      tty: true
```

#### Run Docker
```bash
# one-time init
export MY_UID=$(id -u)
export MY_GID=$(id -g)
docker compose up -d

# open a shell in the SAME persistent container
docker compose exec infrastructure bash

# later
docker compose stop infrastructure     # stop (preserves everything)
docker compose start infrastructure    # restart
docker compose exec infrastructure bash
```

If you encounter **folder permission issues**, run:
```bash
docker run --rm -u 0 -v "[your_folder_path_in_bash]/chiplet":/mnt busybox \
  sh -c "chown -R $MY_UID:$MY_GID /mnt"
```

#### Setup Timeloop for Gemini
Inside Docker:
```bash
cd accelergy-timeloop-infrastructure
make install_accelergy         # install Accelergy
pip3 install ./src/timeloopfe  # install Timeloop Python front-end

# check timeloop availability
which timeloop-mapper || true

# if not installed, build Timeloop
make install_timeloop
```

#### Install XGBoost (C++ API)
Inside Docker:
```bash
git clone --branch v1.6.0 --recursive https://github.com/dmlc/xgboost.git
cd xgboost
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local/xgboost
cmake --build . -j
cmake --install .
```

---

## 🚀 Running Fine-Grained Exploration

### 1. Generate Timeloop Data
```
cd ./TL_predictor/gen_TL_data
python3 run_TL_data_gen.py --wl_type wl
```
- Replace `wl` with one of: `conv_k3`, `conv_k1`, `gemm`.


### 2. Train Timeloop Predictor
```
cd ./TL_predictor/gen_TL_predictor
python3 pred_gen.py -wl_type wl
```

- **XGBoost args**: adjust `num` and `d` (accuracy vs. latency tradeoff).  
- **NN args**: adjust `lr`, `hid_dim`. Default `lr=1e-4` (empirically good; tune per workload).  


### 3. Run Fine-Grained Simulation
1. Copy trained models into `./pred_models`
   - **Remove any filename prefixes** before use.  
2. Run SW-HW co-exploration with different options:
   - **Option 1**: Pure Gemini exploration
   - **Option 2**: Gemini+TLP exploration
   - **Option 3**: Gemini exploration --> w/ Timeloop Inference
   - **Option 4**: Gemini+TLP exploration --> w/ Timeloop Inference

Execute searching with different option (**`mode`** in `.sh` script):
```
make
bash search.sh
```

---

## 🔧 Configurable Parameters

Some important parameters in `.sh` script:

- **`mm`** – Core type  
  - `0`: NVDLA *(not supported in extended experiments)*  
  - `1`: Eyeriss *(supported)*

- **`nn`** – Neural network model ID

  | ID  | Model             | ID  | Model             | ID  | Model             |
  |-----|-------------------|-----|-------------------|-----|-------------------|
  | 0   | darknet19         | 1   | vgg               | 2   | resnet50          |
  | 3   | googlenet         | 4   | resnet101         | 5   | densenet          |
  | 6   | ires              | 7   | gnmt              | 8   | lstm              |
  | 9   | zfnet             | 10  | transformer       | 11  | transformer_cell  |
  | 12  | pnasnet           | 13  | resnext50         | 14  | resnet152         |

  > In extended experiments, **resnet50, vgg, transformer** are used.

- **`bb`** – Batch size  
  - Must be power-of-2 (1, 2, 4, 8, …).  
  - **Max = 64** (limited by extended experiments).

📌 *Changing these parameters allows you to explore different accelerator configurations and workload scenarios. Ensure consistency with memory capacity and architectural constraints.*

---