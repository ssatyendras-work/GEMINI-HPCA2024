# ExploreChiplets

This branch is the original Gemini code ***with technology node from 12nm to 32nm***.

## 🔧 Modifications

1. Updating the technology parameter in `core`, `NoC`, and `NoP`.
2. Migration of L1 cache of Eyeriss to align with the original settings.

---

## 🚀 Run the Baseline Gemini Setup

### 1. Launch Docker Environment

   Run the container with a mounted volume:
   ```bash
   udocker run -v [your-local-path]:/home/workspace/Chiplet4NN timeloop_container
   ```

### 2. Build & Run Simulation

   After launching the container, follow these steps:

   ```bash
   cd ./Chiplet4NN/
   make
   bash [target-tops].sh
   ```

=======
   Replace `[target-tops].sh` with one of the available setup scripts:
   
   - **`72tops.sh`** – Recommended for quick testing (~a few hours of simulation time).
   - **`128tops.sh`** – Intermediate setup (~1–2 days of simulation).
   - **`512tops.sh`** – Full-scale setup (up to one week of simulation time).

   The results should be under `./dse.log/`.

   > **Tip:** Start with `72tops.sh` to validate your environment and configurations before moving on to larger simulations, as the runtime increases exponentially with higher TOPS settings.


---

## ⚙️ Configurable Parameters in Shell Scripts

   Within each `.sh` script, you can adjust the following:
   
   - **`mm`** – Core type:
     - `0`: NVDLA (**not supported in extended experiments**)
     - `1`: Eyeriss (**supported**)
   
   - **`nn`** – Neural network model ID:
      | ID  | Network Model     | ID  | Network Model        | ID  | Network Model        |
      |-----|-------------------|-----|----------------------|-----|----------------------|
      | 0   | darknet19         | 1   | vgg                  | 2   | resnet50             |
      | 3   | googlenet         | 4   | resnet101            | 5   | densenet             |
      | 6   | ires              | 7   | gnmt                 | 8   | lstm                 |
      | 9   | zfnet             | 10  | transformer          | 11  | transformer_cell     |
      | 12  | pnasnet           | 13  | resnext50            | 14  | resnet152            |
     - In the extended experiments, we take resnet50, vgg, and transformer as target models.

   - **`bb`** – Batch size:
     - Should be a power of 2 (e.g., 1, 2, 4, 8, ...).
     - **Maximum allowed batch size is 64**, due to limitations in extended experiments.
   
   > 📌 **Note:** Modifying these parameters allows you to explore different accelerator configurations and workload scenarios. However, ensure consistency with memory capacity and architectural constraints defined in the simulator.
   
   ---
