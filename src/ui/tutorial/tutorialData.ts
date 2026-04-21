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
    id: 'ui-basics',
    title: 'UI Basics — Getting Around',
    description: 'Learn how to navigate NodeTorch: discover tooltips, inspect nodes, and use the dashboard. No model-building yet — just exploring the interface.',
    tasks: [
      {
        id: 'hover-toolbar',
        text: 'Hover over the toolbar buttons',
        hint: 'Every button in the top toolbar has a tooltip explaining what it does. Hover over Run, Train, Test, Infer, Step Through, Export, etc. to discover each one.',
      },
      {
        id: 'help-icons',
        text: 'Hover over a "?" icon in a node\'s properties',
        hint: 'Click any node to open the Property Inspector on the right. Many property labels have a small "?" next to them — hover to read a detailed explanation of what that property does.',
      },
      {
        id: 'keyboard-shortcuts',
        text: 'Open the keyboard shortcuts help',
        hint: 'Press "?" on your keyboard (or Shift+/) to open the shortcuts panel. NodeTorch has shortcuts for undo, copy/paste, select-all, organize, and more.',
      },
      {
        id: 'toggle-dashboard',
        text: 'Toggle the training dashboard',
        hint: 'Press "F" or click the "Dashboard" button at the bottom. The dashboard shows training progress, model summary, hardware info, and compare runs — even before you train.',
      },
      {
        id: 'inspector-learn',
        text: 'Read a node\'s educational description',
        hint: 'Click any node. The inspector on the right shows a "Learn More" paragraph at the top — a plain-English explanation of what that node does and why it matters.',
      },
      {
        id: 'try-layer-detail',
        text: 'Open "View Layer Detail" on any node',
        hint: 'Click a trained node, scroll down in the inspector, and click "View Layer Detail". This opens a modal with weight matrices, feature maps, confusion matrices — rich visualizations specific to each node type.',
        autoDetect: 'layer-detail-opened',
      },
      {
        id: 'toggle-theme',
        text: 'Toggle light/dark theme',
        hint: 'Click the sun/moon icon in the top-right. The whole app switches between light and dark mode. Your choice is saved.',
      },
      {
        id: 'undo-redo',
        text: 'Try undo and redo',
        hint: 'Add a node, then press Ctrl+Z (or Cmd+Z) to undo. Ctrl+Shift+Z to redo. NodeTorch tracks your edit history so you can experiment freely.',
      },
    ],
  },
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
  {
    id: 'vae',
    title: 'VAE — Generative Image Model',
    description: 'A Variational Autoencoder learns a smooth "latent space" it can sample from to generate new images. Unlike a regular autoencoder, the encoder outputs a distribution (mean + variance) instead of a point — enabling interpolation and generation.',
    tasks: [
      {
        id: 'vae-load',
        text: 'Load the "VAE — MNIST" preset',
        hint: 'Click the book icon in the toolbar and pick "VAE — MNIST". This graph has encoder → mean/logvar → Reparameterize → decoder, with a 4-port VAE loss.',
        autoDetect: 'preset-loaded',
      },
      {
        id: 'vae-reparameterize',
        text: 'Click the Reparameterize node and read its description',
        hint: 'This is the heart of a VAE. It takes (mean, logvar) and samples z = mean + exp(0.5*logvar) * N(0,1). The "reparameterization trick" makes sampling differentiable so gradients can flow back through it.',
      },
      {
        id: 'vae-loss-ports',
        text: 'Look at the VAE loss node — it has 4 inputs',
        hint: 'VAE loss = reconstruction loss + KL divergence. It needs (reconstruction, original, mean, logvar). The KL term keeps the latent distribution close to a unit Gaussian, which makes sampling work.',
      },
      {
        id: 'vae-train',
        text: 'Train the VAE',
        hint: 'Click Train. Watch the loss decrease — both reconstruction and KL terms get optimized together. A good VAE has low reconstruction error AND low KL (latent is well-organized).',
        autoDetect: 'training-started',
      },
      {
        id: 'vae-latent-grid',
        text: 'Generate the Latent Space Grid',
        hint: 'After training, click the Reparameterize node → "View Layer Detail" → "Generate Latent Grid". It sweeps across two latent dimensions and decodes each point — showing smooth transitions between digits. Proof the model learned a meaningful latent space.',
        autoDetect: 'layer-detail-opened',
      },
      {
        id: 'vae-step-through',
        text: 'Step through the VAE forward pass',
        hint: 'Open Step Through. You\'ll see the input encoded into mean/logvar, the sampling step, then the decoder reconstructing the image. Compare the reconstruction to the original.',
        autoDetect: 'step-through-opened',
      },
    ],
  },
  {
    id: 'gan',
    title: 'GAN — Adversarial Training',
    description: 'A Generative Adversarial Network pits two networks against each other: a Generator creates fake images from random noise, and a Discriminator tries to tell real from fake. They improve by competing — no reconstruction loss needed.',
    tasks: [
      {
        id: 'gan-load',
        text: 'Load the "GAN — MNIST" preset',
        hint: 'Click the book icon and pick "GAN — MNIST". You\'ll see two separate subgraph blocks (Generator and Discriminator), a Noise Input node, and — unusually — TWO optimizers.',
        autoDetect: 'preset-loaded',
      },
      {
        id: 'gan-noise-input',
        text: 'Look at the Noise Input node',
        hint: 'Unlike other models, the generator\'s input is random noise, not a dataset image. The Noise Input samples a batch of latent vectors each step — this is the "seed" the generator transforms into an image.',
      },
      {
        id: 'gan-dual-opt',
        text: 'Notice the two optimizers',
        hint: 'GANs need alternating updates — one step to train the Discriminator to spot fakes, one step to train the Generator to fool it. Each optimizer gets its own loss port. The training loop handles the alternation automatically.',
      },
      {
        id: 'gan-train',
        text: 'Train the GAN',
        hint: 'Click Train. The dashboard shows both D-Loss and G-Loss curves. Healthy training: they oscillate, neither wins too decisively. If D-Loss → 0, the discriminator has won (generator can\'t fool it). If G-Loss → 0, the discriminator collapsed.',
        autoDetect: 'training-started',
      },
      {
        id: 'gan-samples',
        text: 'Watch generated samples in the dashboard',
        hint: 'Open the "Samples" tab in the training dashboard. Each epoch the generator produces a grid of images — watch them evolve from pure noise into recognizable digits over training.',
      },
    ],
  },
  {
    id: 'diffusion',
    title: 'Diffusion — Iterative Denoising',
    description: 'Diffusion models generate images by iteratively denoising pure noise. During training, they learn to predict the noise added to real images at varying intensity levels. At inference, they reverse the process step by step.',
    tasks: [
      {
        id: 'diff-load',
        text: 'Load the "Diffusion — MNIST" preset',
        hint: 'Click the book icon and pick "Diffusion — MNIST". Notice the Noise Scheduler node — it adds random amounts of noise to images and passes the timestep to the model.',
        autoDetect: 'preset-loaded',
      },
      {
        id: 'diff-scheduler',
        text: 'Inspect the Noise Scheduler node',
        hint: 'This node picks a random timestep per sample, adds proportional noise to the image, and outputs (noisy_image, original_noise, timestep_channel). The model\'s job: given the noisy image + timestep, predict the noise.',
      },
      {
        id: 'diff-timestep-embed',
        text: 'Look at the Timestep Embedding node',
        hint: 'The model needs to know how noisy the input is — a timestep of 1 means "barely noisy", 1000 means "pure noise". Timestep Embedding converts that integer into a rich vector the network can use.',
      },
      {
        id: 'diff-train',
        text: 'Train the Diffusion model',
        hint: 'Click Train. Loss = MSE between predicted noise and actual noise added. It\'s a regression task, not classification — the model just learns to denoise at every intensity level.',
        autoDetect: 'training-started',
      },
      {
        id: 'diff-denoise',
        text: 'Open Step Through and switch to the "Denoise" tab',
        hint: 'After training, Step Through shows a unique Denoise view: starting from pure noise, the model iteratively removes noise step by step until a digit emerges. Scrub through the timeline to watch it materialize.',
        autoDetect: 'step-through-opened',
      },
    ],
  },
  {
    id: 'autoregressive',
    title: 'Language Model — Autoregressive Generation',
    description: 'A character-level language model predicts the next character given previous characters. Train it on Shakespeare and generate new text one character at a time. This is the same principle behind GPT — just smaller and character-level.',
    tasks: [
      {
        id: 'ar-load',
        text: 'Load the "Char-LM — Tiny Shakespeare" preset',
        hint: 'Click the book icon and pick "Char-LM — Tiny Shakespeare". The graph: data → Embedding → LSTM → Linear → CrossEntropy → Adam. The Shakespeare corpus (~1MB) gets chunked into sequences for training.',
        autoDetect: 'preset-loaded',
      },
      {
        id: 'ar-embedding',
        text: 'Click the Embedding node',
        hint: 'Text can\'t go straight into a neural net — characters are discrete integer IDs. The Embedding layer maps each character ID (0-64 for 65 unique chars) to a learnable dense vector, giving the model a continuous representation.',
      },
      {
        id: 'ar-train',
        text: 'Train the language model',
        hint: 'Click Train. This takes longer than classification — LSTMs are sequential. Watch the dashboard: the "Perplexity" tab shows how uncertain the model is per character (lower = better). At each epoch you\'ll also see generated text samples.',
        autoDetect: 'training-started',
      },
      {
        id: 'ar-perplexity',
        text: 'Open the Perplexity tab in the dashboard',
        hint: 'Perplexity = exp(cross-entropy loss). A random model has perplexity ≈ 65 (the vocab size). A well-trained model drops below 5 — meaning it effectively picks between ~5 likely next characters instead of all 65.',
      },
      {
        id: 'ar-generate',
        text: 'Generate text in the Step Through panel',
        hint: 'After training, open Step Through → "Generate" tab. Type a prompt (or leave empty), adjust Temperature (controls randomness — lower = more deterministic), and click Generate. Watch the model produce Shakespeare-like text one character at a time.',
        autoDetect: 'step-through-opened',
      },
    ],
  },
];

/** All task IDs in order */
export const ALL_TASK_IDS = TUTORIAL_GOALS.flatMap(g => g.tasks.map(t => t.id));
