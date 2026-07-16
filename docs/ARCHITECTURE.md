# BankGraphAI — Graph Architecture Document

## 1. Mathematical Modeling of the Graph

### 1.1 Node Definition

Let $\mathcal{D} = \{r_1, r_2, \ldots, r_N\}$ be the dataset of $N$ accounting records (rows).

Each record $r_i$ becomes exactly one node $v_i \in V$ in the graph $G = (V, E)$.

**Justification:** Each accounting entry is an independent financial event. Treating each row as a node preserves the granularity of the original data and allows us to model relationships between entries based on their feature similarity.

### 1.2 Feature Definition

Each node $v_i$ has a feature vector $\mathbf{x}_i \in \mathbb{R}^d$ derived from its row $r_i$.

Let the columns of $\mathcal{D}$ be partitioned into:

- **Numerical features** $\mathcal{C}_{\text{num}}$: continuous or discrete numeric values (e.g., movement value, tax amounts)
- **Categorical features** $\mathcal{C}_{\text{cat}}$: string identifiers (e.g., company, partner, account)
- **Text features** $\mathcal{C}_{\text{text}}$: descriptive text (e.g., description, name)
- **Temporal features** $\mathcal{C}_{\text{time}}$: dates and timestamps

#### Feature Engineering Pipeline

For each node $v_i$:

1. **Numerical encoding**: For each $c \in \mathcal{C}_{\text{num}}$, apply z-score normalization:
   $$\tilde{x}_{i,c} = \frac{x_{i,c} - \mu_c}{\sigma_c + \epsilon}$$
   where $\mu_c$ and $\sigma_c$ are the column mean and standard deviation.

2. **Categorical encoding**: For each $c \in \mathcal{C}_{\text{cat}}$ with cardinality $K_c$, apply one-hot encoding or hash encoding:
   - If $K_c \leq 20$: one-hot encoding → $K_c$ binary features
   - If $K_c > 20$: hash encoding → $\lfloor \log_2 K_c \rfloor + 1$ features via $h(x) = \text{hash}(x) \mod M$

3. **Text encoding**: For each $c \in \mathcal{C}_{\text{text}}$, use SentenceTransformer to produce a 384-dimensional embedding.

4. **Temporal encoding**: Decompose dates into cyclic features:
   $$t_{\text{sin}} = \sin\left(\frac{2\pi \cdot \text{day}}{365}\right), \quad t_{\text{cos}} = \cos\left(\frac{2\pi \cdot \text{day}}{365}\right)$$

The final feature vector is the concatenation:
$$\mathbf{x}_i = [\mathbf{x}_i^{\text{num}} \parallel \mathbf{x}_i^{\text{cat}} \parallel \mathbf{x}_i^{\text{text}} \parallel \mathbf{x}_i^{\text{time}}]$$

**Justification:** This preserves all information from the original row while making it suitable for distance computation. Numerical features are normalized to prevent scale dominance. Categorical features are encoded to capture grouping structure. Text features capture semantic similarity. Temporal features capture seasonality.

### 1.3 Edge Definition

Edges are **not** given by the data. They are **computed** from feature similarity.

#### 1.3.1 Similarity Metrics

For any two nodes $v_i, v_j$ with feature vectors $\mathbf{x}_i, \mathbf{x}_j$:

**Cosine Similarity** (for high-dimensional sparse features):
$$s_{\text{cos}}(i,j) = \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\|_2 \|\mathbf{x}_j\|_2}$$

**Euclidean Distance** (for dense numerical features):
$$d_{\text{euc}}(i,j) = \|\mathbf{x}_i - \mathbf{x}_j\|_2$$

**Gaussian RBF Kernel** (bounded similarity):
$$s_{\text{rbf}}(i,j) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|_2^2}{2\sigma^2}\right)$$

**Attribute Sharing** (for categorical agreement):
$$s_{\text{attr}}(i,j) = \frac{1}{|\mathcal{C}_{\text{cat}}|} \sum_{c \in \mathcal{C}_{\text{cat}}} \mathbb{1}[x_{i,c} = x_{j,c}]$$

#### 1.3.2 Edge Construction Strategies

**Strategy A: k-Nearest Neighbors (kNN)**
For each node $v_i$, connect to its $k$ most similar nodes:
$$E_{\text{kNN}} = \{(i,j) : j \in \text{NN}_k(i)\}$$
where $\text{NN}_k(i)$ are the $k$ nodes with highest $s_{\text{cos}}(i,j)$.

*Justification:* kNN creates a graph where each node is connected to its most similar peers. This is computationally efficient ($O(N \log N)$ with approximate nearest neighbors) and produces a sparse, connected graph suitable for community detection.

**Strategy B: $\epsilon$-Neighborhood Graph**
Connect nodes whose similarity exceeds a threshold $\tau$:
$$E_{\epsilon} = \{(i,j) : s_{\text{cos}}(i,j) > \tau\}$$

*Justification:* This produces a graph where edge density reflects the natural clustering structure. The threshold $\tau$ can be set as the $p$-th percentile of all pairwise similarities.

**Strategy C: Shared Attribute Graph**
Connect nodes that share a critical attribute (e.g., same company, same partner):
$$E_{\text{shared}} = \{(i,j) : \exists c \in \mathcal{C}_{\text{critical}} \text{ with } x_{i,c} = x_{j,c}\}$$

*Justification:* In accounting data, entries sharing the same company or partner are inherently related. This creates a graph based on business logic rather than statistical similarity.

**Strategy D: Heterogeneous Multi-Edge Graph**
Create multiple edge types, each from a different strategy:
- Type 1: Cosine similarity kNN edges
- Type 2: Shared company edges
- Type 3: Shared partner edges
- Type 4: Temporal proximity edges (entries within the same week)

This produces a **heterogeneous graph** $G = (V, E_1, E_2, \ldots, E_m)$ where each edge type captures a different relationship.

*Justification:* Financial transactions have multiple orthogonal relationships. A heterogeneous graph captures this richness and allows algorithms to leverage different relationship types.

### 1.4 Graph Construction Algorithm

For scalability to millions of records, we use a **hybrid approach**:

1. **Small data** ($N < 100,000$): Compute full similarity matrix, apply kNN or $\epsilon$-threshold
2. **Medium data** ($N < 1,000,000$): Use approximate nearest neighbors (Annoy, FAISS) with cosine similarity
3. **Large data** ($N \geq 1,000,000$): Use PySpark + GraphFrames with distributed computation

#### Algorithm: Scalable Graph Construction

```
Input: Feature matrix X ∈ ℝ^{N×d}, strategy S, parameters k, τ
Output: Edge list E

1. If N < 100,000:
   a. Compute similarity matrix S = X · X^T (normalized)
   b. For each row i, find top-k similar nodes
   c. Add edges (i, j) for each neighbor j

2. If N < 1,000,000:
   a. Build Annoy index with cosine distance
   b. For each node i, query k nearest neighbors
   c. Add edges (i, j) for each neighbor j

3. If N ≥ 1,000,000:
   a. Use PySpark to partition data
   b. For each partition, compute local kNN
   c. Use GraphFrames to construct distributed graph
   d. Apply Motif finding for cross-partition edges
```

### 1.5 Community Detection

Given graph $G = (V, E)$, we detect communities using multiple algorithms:

#### Modularity Maximization (Louvain / Leiden)

Maximize modularity $Q$:
$$Q = \frac{1}{2m} \sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2m}\right] \delta(c_i, c_j)$$

where $A_{ij}$ is the adjacency matrix, $k_i$ is the degree of node $i$, $m$ is the total number of edges, and $\delta(c_i, c_j) = 1$ if nodes $i$ and $j$ are in the same community.

**Leiden algorithm** improves on Louvain by guaranteeing connected communities and faster convergence.

#### Label Propagation

Each node adopts the most frequent label among its neighbors:
$$c_i^{(t+1)} = \arg\max_c \sum_{j \in \mathcal{N}(i)} \mathbb{1}[c_j^{(t)} = c]$$

*Justification:* Linear time complexity $O(m)$, suitable for large graphs.

#### Spectral Clustering

Use the eigenvectors of the graph Laplacian $L = D - A$ to embed nodes in $\mathbb{R}^k$, then apply k-means.

*Justification:* Theoretically optimal for the normalized cut objective. Works well when communities are well-separated.

#### Infomap

Minimizes the map equation:
$$L(M) = q_{\curvearrowright} H(\mathcal{Q}) + \sum_{\alpha=1}^m p_{\circlearrowright}^\alpha H(\mathcal{P}^\alpha)$$

*Justification:* Information-theoretic approach that captures flow patterns in the graph.

### 1.6 Node Embeddings

For downstream tasks (visualization, anomaly detection), we generate node embeddings:

#### Node2Vec

Maximizes the likelihood of preserving node neighborhoods in a low-dimensional space:
$$\max_f \sum_{u \in V} \log \Pr(N_S(u) \mid f(u))$$

where $N_S(u)$ is the neighborhood of $u$ under sampling strategy $S$ (biased random walk).

#### GraphSAGE

Inductive embedding generation:
$$\mathbf{h}_v^{(k)} = \sigma\left(W_k \cdot \text{AGGREGATE}_k\left(\{\mathbf{h}_u^{(k-1)}, \forall u \in \mathcal{N}(v)\}\right)\right)$$

*Justification:* Can generate embeddings for unseen nodes without retraining.

#### GCN

Graph Convolutional Network:
$$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}\mathbf{H}^{(l)}W^{(l)}\right)$$

### 1.7 Community Explainability

For each community $C$, we compute:

1. **Feature importance**: For each feature $f$, compute the intra-community mean vs global mean:
   $$\Delta_f(C) = \left|\mu_f^{(C)} - \mu_f^{(\text{global})}\right|$$

2. **Top distinguishing features**: The $k$ features with largest $|\Delta_f(C)|$

3. **Community profile**: Statistical summary of the community:
   - Mean, std, min, max of numerical features
   - Mode of categorical features
   - Most frequent text terms

4. **Node-level explanation**: For each node $v \in C$, the features that most strongly influenced its assignment:
   $$\text{influence}_f(v) = \frac{|x_{v,f} - \mu_f^{(C)}|}{\sigma_f^{(C)} + \epsilon}$$

---

## 2. Implementation Architecture

### Module Structure

```
app/
├── graph_engine/              # NEW: Complete graph engine
│   ├── __init__.py
│   ├── node_features.py       # Feature extraction & normalization
│   ├── edge_builder.py        # Edge construction strategies
│   ├── graph_factory.py       # Orchestrates graph construction
│   ├── community.py           # Community detection (all algorithms)
│   ├── embeddings.py          # Node2Vec, DeepWalk, GraphSAGE, GCN
│   ├── explainer.py           # Community & node explainability
│   └── visualization.py       # PyVis, Plotly, NetworkX rendering
├── dashboard.py               # Updated dashboard
└── ... (existing files)
```

### Data Flow

```
CSV → pandas/Spark → Feature Vectors → Similarity Matrix → Edge List → Graph
                                                                          ↓
                                                              Community Detection
                                                                          ↓
                                                              Node Embeddings
                                                                          ↓
                                                          Explainability + Viz
```

### Scalability Strategy

| Dataset Size | Engine | Edge Strategy | Community Detection |
|---|---|---|---|
| < 100K rows | pandas + sklearn | Full kNN | All algorithms |
| 100K–1M | pandas + Annoy/FAISS | Approximate kNN | Louvain, Leiden, LP |
| > 1M | PySpark + GraphFrames | Distributed kNN | Label Propagation |