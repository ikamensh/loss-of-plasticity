# Drifting Class Sampling Refactor

## Architecture Options

### Option A: Standalone Sampler Class
- **Description:** Provide a `DriftingClassSampler` utility that maintains class weights, drifts them, and emits class indices. Training scripts supply dataset-specific logic.
- **Pros:**
  - Decouples sampling mechanics from dataset specifics.
  - Easy to unit test by inspecting class choices and weight dynamics.
  - Keeps training loop explicit and readable.
- **Cons:**
  - Requires a small amount of integration code in each script that uses it.

### Option B: PyTorch `Sampler` Subclass
- **Description:** Implement a custom iterator compatible with `DataLoader` that yields indices according to drifting weights.
- **Pros:**
  - Plugs directly into `DataLoader`, eliminating manual batching code.
  - Can be reused across multiple datasets without changes.
- **Cons:**
  - More moving parts—needs to respect epoch boundaries and worker process semantics.
  - Harder to test because sampling is hidden inside the `DataLoader` machinery.

### Option C: Dataset Wrapper
- **Description:** Create a `DriftingMNIST` dataset that encapsulates MNIST along with drifting class logic in its `__getitem__`.
- **Pros:**
  - Keeps training loops identical to stationary datasets.
  - Encapsulates all state within the dataset object.
- **Cons:**
  - Conflates data storage with sampling strategy, making reuse for other datasets awkward.
  - Harder to unit test in isolation because it must expose dataset internals.

## Decision
Option A was chosen for its clarity and testability. A standalone sampler keeps the
random-walk logic isolated while allowing training scripts to remain simple and
explicit. It introduces minimal complexity and can serve as a building block for
future DataLoader-based implementations if needed.

