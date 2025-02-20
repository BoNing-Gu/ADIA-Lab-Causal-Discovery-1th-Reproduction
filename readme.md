# Introduction

This Repo is still on progress  
See the 1th solution here: https://thetourney.github.io/adia-report/

# Data Processing

The core data processing module of this project constructs dynamic relationship matrices between features using kernel regression. This section focuses on explaining the principles and implementation details of building the third channel (i.e., the regression coefficient channel) during data preprocessing.

## Channel Structure Description

The preprocessed tensor has dimensions `[n_edges, 3, n_samples]`, where:

- **Channel 0**: Source feature data (u)
- **Channel 1**: Target feature data (v)
- **Channel 2**: Influence coefficients (c)

## Third Channel Construction Process

1. **Feature Relationship Enumeration**
   - Generate a full permutation edge set for p-dimensional features: `edges = [(u, v) | u,v ∈ [0,p-1], u≠v]`
   - A total of p×(p-1) directed edges are generated.

2. **Kernel Regression Modeling**
   - For each target feature v:
     - Use feature v as the dependent variable: `endog = X[:, v]`
     - Use the remaining p-1 features as independent variables: `exog = X[:, u≠v]`
     - Perform local linear regression using a Gaussian kernel:

       ```python
       model = KernelReg(endog, exog, var_type='c'*(p-1), reg_type='ll', bw=[0.3]*(p-1))
       ```

3. **Dynamic Coefficient Extraction**
   - Obtain the local regression coefficient matrix after fitting: `coeffs_matrix = model.fit()[1]`
   - The matrix has dimensions `[n_samples, p-1]`, recording the influence of features on the target feature.
   - Map the corresponding coefficients to the third channel of the edge set:

     ```python
     preprocessed[edge_idx, 2, :] = coeffs_matrix[:, u_idx]
     ```

## Visualization Method

Use the `plot_preprocessed` function to observe feature relationships:

```python
plot_preprocessed(df, preprocessed, 'X1', 'X2')
```

This will generate two plots:

1. Scatter plot of X1 vs. X2
2. Relationship plot of X1 vs. dynamic coefficients

# Model Construction

## From preprocessed data to CausalDataset

The next step in the project involves constructing a dataset that combines the preprocessed data and labels, which is used for causal learning tasks. The CausalDataset class facilitates this by handling the edge features, node labels, and edge labels necessary for training causal models. In this section, we detail how the data is structured and the underlying preprocessing that allows for training models to learn causal relationships between the features.

### Data Loading and Preprocessing

The preprocessed data, once created, is used to create the CausalDataset. The preprocessing steps are handled by the preprocess_dict function, which outputs processed tensors that include the dynamic relationship coefficients and relevant feature data. The dataset is loaded and used for training the causal model, which is typically represented by a directed acyclic graph (DAG).

### Dataset Overview
The CausalDataset class is a custom PyTorch dataset that loads and processes data samples from a dictionary structure. Each sample corresponds to a set of features (X), their preprocessed version (X_processed), and the corresponding adjacency matrix (Y) that encodes causal relationships.

1. Edge Features:

- These represent the dynamic coefficients between pairs of features that express the causal influence of one feature on another.
- The features are stored in a tensor with dimensions [n_edges, 3, n_samples], where the three channels include the source and target feature data, and the regression coefficients.

2. Edge Types:

- Each edge type indicates a specific relationship (e.g., cause, consequence, mediator, confounder, etc.).
- These types are encoded as integers, representing different types of causal relationships between nodes.

3. Node Labels:

- Each node (feature) is labeled according to its causal role in the graph, which could be a confounder, mediator, cause, or consequence. These labels are stored in a one-hot encoded matrix.

4. Edge Labels:

- Each edge in the graph is labeled as either a causal (1) or non-causal (0) relationship based on the adjacency matrix from the true causal graph.

## Model Architecture

The architecture of the `CausalModel` is designed to learn and predict causal relationships between features based on a directed acyclic graph (DAG). It utilizes a combination of convolutional layers, self-attention mechanisms, and linear transformations to process edge features and node relationships. Below is a detailed breakdown of the model components:

### ConvBlock 

The `ConvBlock` class implements a residual convolutional block that processes feature data through convolutional layers, normalization, and activation functions:

- **Convolution Layer**: A 1D convolution (`Conv1d`) is used to capture local feature interactions across different edge features.
- **Group Normalization**: Applied to normalize the activations and stabilize training.
- **Activation**: GELU activation function is used for nonlinearities.
- **Residual Connection**: A skip connection is added from the input to the output to help prevent vanishing gradients and improve model convergence.

### SelfAttentionBlock

The `SelfAttentionBlock` class implements multi-head self-attention:

- **Multihead Attention**: This block allows the model to learn dependencies between features from different perspectives by attending to different parts of the input simultaneously.
- **Layer Normalization**: Applied to normalize the attention outputs, enhancing stability and reducing the risk of overfitting.

### 3. MergeBlock

The `MergeBlock` is responsible for merging feature embeddings from multiple edge indices:

- **Linear Transformation**: The concatenated embeddings are passed through a linear layer to reduce the dimensionality.
- **Layer Normalization**: Applied after the linear transformation to standardize the input.
- **Activation**: The GELU activation function adds non-linearity to the transformation.

### 4. CausalModel

The `CausalModel` class ties together the various blocks and processes the input data in multiple stages:

- **Stem Layer**: A simple 1D convolution to initialize the edge feature representation.
  
- **Convolutional Blocks**: A series of `ConvBlock` layers are used to capture intricate relationships between features. The model processes edge features in this stage, improving its representation of dynamic coefficients and feature dependencies.

- **Pooling Layer**: An adaptive average pooling layer is used to aggregate information across time/sequence dimensions, reducing the dimensionality.

- **Edge Type Embedding**: An embedding layer maps edge type indices to a fixed-size hidden vector, which is added to the processed edge feature data to introduce information about the relationship type between features.

- **Self-Attention Layers**: Multiple `SelfAttentionBlock` layers allow the model to learn complex dependencies between edge features, making it more adept at recognizing causal relationships.

- **Edge Classification**: A fully connected layer (`edge_cls`) is applied to the output of the attention layers to classify whether a given edge represents a causal or non-causal relationship.

- **Node Classification**: The `node_merge` block merges the embeddings of each node, and a final linear layer (`node_cls`) classifies the role of each node in the causal graph (e.g., confounder, mediator, cause, consequence).

### Forward Pass

The forward pass takes a batch of data containing edge features, edge types, and variable names. It extracts node-level and edge-level embeddings and performs the following steps:

- **Feature Extraction**: The edge features are passed through the stem and convolutional blocks to generate learned feature representations.
- **Edge Type Embedding**: The edge type information is incorporated into the feature representation.
- **Self-Attention**: The features undergo attention-based transformations to capture complex relationships between edges.
- **Edge Classification**: The model classifies the edges as causal or non-causal.
- **Node Classification**: For each node (excluding the target variables `x_var` and `y_var`), the model predicts its causal role in the graph using the merged embeddings from neighboring nodes and edges.

### Model Output

- **Edge Logits**: The model returns edge-level logits, which represent the probability that each edge is causal (1) or non-causal (0).
- **Node Logits**: The model also returns node-level logits, which correspond to the predicted causal role of each node (confounder, mediator, etc.).

This architecture is designed to process dynamic, graph-structured data, leveraging convolutional layers, attention mechanisms, and embeddings to learn and predict complex causal relationships.

## Model Training

The training process for the `CausalModel` is organized using PyTorch Lightning, which simplifies the workflow. Here's a streamlined overview of the key components:

### CausalDataModule

The `CausalDataModule` handles data loading for both training and validation:

- **train_dataloader**: Loads the training data with shuffling and parallel processing.
- **val_dataloader**: Loads the validation data without shuffling.

### CausalLightningModule

The `CausalLightningModule` wraps the `CausalModel` for training and validation:

- **Forward Pass**: Defines how the model processes each batch of data.
- **Loss and Metrics**: Computes loss and accuracy for both edges and nodes.
- **Optimizer**: Uses the AdamW optimizer with a cosine annealing learning rate scheduler.

### Class Weights

Class weights for edge and node labels are calculated to address class imbalance. These weights help adjust the loss function, giving more importance to underrepresented classes.

### Training Process

The training loop is controlled by the `Trainer`:

- **Data Splitting**: The data is split into training and test sets.
- **Model and Optimizer Setup**: The model is initialized, and class weights are computed.
- **Training Configuration**: The model is trained for up to 50 epochs using mixed precision and checkpointing.

### Model Evaluation

During training, the model's performance is evaluated on the validation set. Checkpoints allow the model to be saved and resumed from any epoch. After training, the best model is selected based on validation metrics.

This setup ensures efficient training and evaluation while handling class imbalance and enabling easy checkpointing.