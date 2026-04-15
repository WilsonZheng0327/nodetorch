# Future Ideas

Features to consider once the core layers are working.

## Step Execution
Run the graph one node at a time instead of all at once. Lets users pause after each layer, inspect activations, then continue. Useful for understanding data flow.

The engine already walks nodes sequentially in a `for` loop — converting to a generator or step-based API would be straightforward. Not needed until the Python backend is running forward passes.

## Breakpoints
Mark specific nodes as pause points. Engine runs until it hits one, then yields control. Pairs with step execution.

## Backward Pass Visualization
PyTorch handles backpropagation internally via autograd — we don't control it step-by-step. After `loss.backward()` runs, the backend sends per-node gradient info (shapes, magnitudes, histograms) back to the frontend for visualization. Users see what happened at each layer even though the backward pass ran as one unit.
