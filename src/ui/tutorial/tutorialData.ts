/** Tutorial goals and tasks — builds a small CNN for CIFAR-10 step by step.
 *
 *  Each goal is a milestone. Tasks within a goal are ordered steps.
 *  Some tasks have an `autoDetect` key — the tutorial panel listens for
 *  these events and marks them complete automatically.
 */

export interface TutorialTask {
  id: string;
  text: string;
  hint?: string;        // extra help shown on hover/expand
  autoDetect?: string;  // event key that auto-completes this task
}

export interface TutorialGoal {
  id: string;
  title: string;
  description: string;
  tasks: TutorialTask[];
}

export const TUTORIAL_GOALS: TutorialGoal[] = [
  {
    id: 'build',
    title: 'Build a CNN for CIFAR-10',
    description: 'Build a small convolutional neural network that classifies 32x32 color images into 10 categories (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).',
    tasks: [
      {
        id: 'open-palette',
        text: 'Open the node palette',
        hint: 'Press Tab or click the "Nodes" button in the top-left corner.',
        autoDetect: 'palette-opened',
      },
      {
        id: 'add-data',
        text: 'Add a CIFAR-10 dataset node',
        hint: 'In the palette, expand Data > Image and drag "CIFAR-10" onto the canvas. This provides 32x32 RGB images and their labels.',
        autoDetect: 'node-added-data',
      },
      {
        id: 'add-conv',
        text: 'Add a Conv2d layer',
        hint: 'From ML > Layers > Convolution, drag "Conv2d" onto the canvas. Set outChannels to 32, kernelSize to 3, padding to 1. Connect CIFAR-10 "Images" output → Conv2d "Input".',
        autoDetect: 'node-added-conv2d',
      },
      {
        id: 'add-relu',
        text: 'Add a ReLU activation',
        hint: 'From ML > Activations, drag "ReLU". Connect Conv2d output → ReLU input. ReLU zeros out negative values — this non-linearity lets the network learn complex patterns.',
        autoDetect: 'node-added-activation',
      },
      {
        id: 'add-pool',
        text: 'Add a MaxPool2d layer',
        hint: 'From ML > Layers > Pooling, drag "MaxPool2d". Set kernelSize to 2, stride to 2. This halves the spatial dimensions (32x32 → 16x16), keeping the strongest features.',
        autoDetect: 'node-added-pool',
      },
      {
        id: 'add-flatten',
        text: 'Add a Flatten layer',
        hint: 'From ML > Layers, drag "Flatten". Connect MaxPool output → Flatten input. This converts the 3D feature maps [C, H, W] into a 1D vector that Linear layers can process.',
        autoDetect: 'node-added-flatten',
      },
      {
        id: 'add-linear',
        text: 'Add a Linear layer (set outFeatures to 10)',
        hint: 'From ML > Layers, drag "Linear". Set outFeatures to 10 (one per CIFAR-10 class). Connect Flatten output → Linear input.',
        autoDetect: 'node-added-linear',
      },
      {
        id: 'add-loss',
        text: 'Add a CrossEntropy loss node',
        hint: 'From ML > Loss, drag "CrossEntropy". It has TWO inputs: connect Linear output → "Predictions" port, and CIFAR-10 "Labels" output → "Labels" port. The loss measures how wrong the predictions are.',
        autoDetect: 'node-added-loss',
      },
      {
        id: 'add-optimizer',
        text: 'Add an Adam optimizer',
        hint: 'From ML > Optimizers, drag "Adam". Connect CrossEntropy output → Adam "Loss" port. The optimizer adjusts weights to minimize the loss. Set epochs to 5 to start.',
        autoDetect: 'node-added-optimizer',
      },
      {
        id: 'connect-all',
        text: 'Connect all nodes in a chain',
        hint: 'Make sure every node is connected: CIFAR-10 → Conv2d → ReLU → MaxPool → Flatten → Linear → CrossEntropy → Adam. Also connect CIFAR-10 Labels → CrossEntropy Labels.',
        autoDetect: 'edge-added',
      },
      {
        id: 'run-shape',
        text: 'Click Run to verify shapes',
        hint: 'Click the green "Run" button in the toolbar. Every node should show output shapes without errors. If you see red, check your connections and property values.',
        autoDetect: 'forward-run',
      },
    ],
  },
  {
    id: 'train',
    title: 'Train & Evaluate',
    description: 'Train the model on CIFAR-10 and test its predictions.',
    tasks: [
      {
        id: 'start-train',
        text: 'Click Train to start training',
        hint: 'Click the "Train" button. The training dashboard opens automatically showing live loss and accuracy curves. Watch the loss go down!',
        autoDetect: 'training-started',
      },
      {
        id: 'check-dashboard',
        text: 'Explore the training dashboard',
        hint: 'Press F to toggle the dashboard. Check the Loss tab (should decrease), Accuracy tab (should increase), and Gradients tab (shows gradient health per layer).',
      },
      {
        id: 'check-epochs',
        text: 'Open the Epochs tab',
        hint: 'Click "Epochs" in the dashboard tabs. This table shows per-epoch metrics: loss, accuracy, validation accuracy, learning rate, and training time.',
      },
      {
        id: 'run-infer',
        text: 'Click Infer to test on a new sample',
        hint: 'After training finishes, click "Infer". The CIFAR-10 node shows the input image with its actual label, and the Linear node shows the predicted class.',
        autoDetect: 'infer-run',
      },
    ],
  },
  {
    id: 'inspect',
    title: 'Inspect & Understand',
    description: 'Use visualization tools to understand what the model learned.',
    tasks: [
      {
        id: 'step-through',
        text: 'Open Step Through (footprints icon)',
        hint: 'Click the footprints icon in the toolbar. This loads a sample and walks through each layer, showing how the input image transforms into a prediction.',
        autoDetect: 'step-through-opened',
      },
      {
        id: 'step-backward',
        text: 'Switch to the Backward tab in Step Through',
        hint: 'Click the "Backward" tab at the top of the step-through panel. This shows how gradients flow backward — the learning signal that updates each layer\'s weights.',
      },
      {
        id: 'layer-detail',
        text: 'Open layer detail on the loss node',
        hint: 'Click the CrossEntropy node, then click "Layer Detail" in the inspector. View the confusion matrix to see which classes get confused with each other.',
        autoDetect: 'layer-detail-opened',
      },
      {
        id: 'toggle-viz',
        text: 'Toggle a viz panel on the Conv2d node',
        hint: 'Click the eye icon on the Conv2d node. This shows live weight distributions, gradient histograms, and health indicators (vanishing/exploding gradient warnings).',
        autoDetect: 'viz-toggled',
      },
      {
        id: 'try-preset',
        text: 'Try loading a different preset',
        hint: 'Click the book icon in the toolbar to open the preset browser. Try "VGG-lite — CIFAR-10" for a deeper CNN, or "LSTM — IMDb" for a text classification model!',
        autoDetect: 'preset-loaded',
      },
    ],
  },
];

/** All task IDs in order */
export const ALL_TASK_IDS = TUTORIAL_GOALS.flatMap(g => g.tasks.map(t => t.id));
