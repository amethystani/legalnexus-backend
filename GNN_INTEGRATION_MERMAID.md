# GNN Integration Flow - Mermaid Diagrams

## Complete System Integration

```mermaid
flowchart TD
    Start([Start: Legal Documents]) --> Load[Load Legal Documents<br/>JSON files + CSV data]
    
    Load --> CreateNodes[Create Nodes in Neo4j]
    CreateNodes --> CaseNode[Case nodes<br/>id, title, court, date, text]
    CreateNodes --> JudgeNode[Judge nodes<br/>name]
    CreateNodes --> CourtNode[Court nodes<br/>name]
    CreateNodes --> StatuteNode[Statute nodes<br/>name]
    
    CaseNode --> CreateRel[Create Relationships]
    JudgeNode --> CreateRel
    CourtNode --> CreateRel
    StatuteNode --> CreateRel
    
    CreateRel --> JudgedRel[Judge -JUDGED→ Case]
    CreateRel --> HeardRel[Case -HEARD_BY→ Court]
    CreateRel --> RefRel[Case -REFERENCES→ Statute]
    
    JudgedRel --> GenEmbed[Generate Embeddings<br/>Gemini API]
    HeardRel --> GenEmbed
    RefRel --> GenEmbed
    
    GenEmbed --> VectorEmbed[Convert text → 768-dim vectors<br/>Store in Case.embedding]
    VectorEmbed --> CreateIndex[Create Vector Index<br/>vector_index + entity_index]
    
    CreateIndex --> Neo4jGraph[(Neo4j Knowledge Graph<br/>Nodes + Relationships + Indexes)]
    
    Neo4jGraph --> GNNExtract[GNN: Extract Graph Data]
    Neo4jGraph --> VectorSearch[Vector Index Search]
    
    GNNExtract --> Query1[Query 1: Get all nodes<br/>MATCH n RETURN id, labels, properties]
    GNNExtract --> Query2[Query 2: Get all relationships<br/>MATCH source-r→target]
    
    Query1 --> CreateFeatures[Create Features<br/>9-dim vectors per node]
    Query2 --> EdgeIndex[Create edge_index<br/>source→target pairs]
    
    CreateFeatures --> GNNTrain[GNN Training]
    EdgeIndex --> GNNTrain
    
    GNNTrain --> FeatureEng[Feature Engineering 9-dim]
    FeatureEng --> GCN1[GCN Layer 1: 9→64]
    GCN1 --> ReLU1[ReLU + Dropout 0.5]
    ReLU1 --> GCN2[GCN Layer 2: 64→64]
    GCN2 --> LinkHead[Link Prediction Head<br/>Concatenate + Dense layers]
    LinkHead --> TrainResult[Training Result<br/>ROC-AUC: 0.78-0.85]
    
    TrainResult --> Predictions[GNN Predictions]
    Predictions --> Pred1[Case_123 -CITES→ Case_456<br/>89% probability]
    Predictions --> Pred2[Case_789 -REFERENCES→ Statute_X<br/>92% probability]
    Predictions --> Pred3[Judge_Y -JUDGED→ Case_321<br/>85% probability]
    
    Pred1 --> Integration[Integration with Search]
    Pred2 --> Integration
    Pred3 --> Integration
    VectorSearch --> Integration
    
    Integration --> UserQuery[User Query:<br/>Section 65B electronic evidence]
    UserQuery --> VectorStep[Step 1: Vector Index Search<br/>Gemini 768-dim embeddings]
    UserQuery --> GNNStep[Step 2: GNN Enhancement<br/>Predicted relationships]
    
    VectorStep --> CombineResults[Combined Results]
    GNNStep --> CombineResults
    
    CombineResults --> FinalResult[Direct matches Vector<br/>+ Related cases GNN<br/>+ Graph relationships]
    
    style Neo4jGraph fill:#E6F3FF,stroke:#2E86AB,stroke-width:3px
    style GNNTrain fill:#FFF4E6,stroke:#F77F00,stroke-width:3px
    style Predictions fill:#E6FFE6,stroke:#06A77D,stroke-width:3px
    style Integration fill:#F0E6FF,stroke:#6C5CE7,stroke-width:3px
    style FinalResult fill:#D4EDDA,stroke:#28A745,stroke-width:3px
```

---

## Simplified GNN Architecture Flow

```mermaid
graph TB
    subgraph Input["INPUT LAYER"]
        Graph[Legal Case Graph<br/>Cases, Judges, Courts, Statutes]
    end
    
    subgraph Features["FEATURE ENGINEERING"]
        F1[Node Type: Case/Judge/Court/Statute<br/>4 features: one-hot encoded]
        F2[Title Length: normalized<br/>1 feature]
        F3[Court Hierarchy: SC/HC/District<br/>3 features: one-hot encoded]
        F4[Year: 1950-2020 normalized<br/>1 feature]
        F5[Total: 9-dimensional vector]
    end
    
    subgraph GNN["GNN ARCHITECTURE"]
        GCN1[GCN Layer 1<br/>Input: 9-dim → Output: 64-dim<br/>Aggregates neighbor info]
        Activation1[ReLU Activation<br/>max0, x<br/>+ Dropout 0.5]
        GCN2[GCN Layer 2<br/>Input: 64-dim → Output: 64-dim<br/>Higher-level patterns]
    end
    
    subgraph Prediction["LINK PREDICTION"]
        Concat[Concatenate Embeddings<br/>Source 64 + Target 64 = 128]
        Dense1[Dense Layer: 128 → 64<br/>+ ReLU + Dropout]
        Dense2[Dense Layer: 64 → 1<br/>+ Sigmoid]
        Output[Probability: 0.0 to 1.0<br/>Relationship likelihood]
    end
    
    Graph --> F1
    Graph --> F2
    Graph --> F3
    Graph --> F4
    F1 --> F5
    F2 --> F5
    F3 --> F5
    F4 --> F5
    
    F5 --> GCN1
    GCN1 --> Activation1
    Activation1 --> GCN2
    GCN2 --> Concat
    Concat --> Dense1
    Dense1 --> Dense2
    Dense2 --> Output
    
    style Graph fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    style F5 fill:#FFF9C4,stroke:#F57C00,stroke-width:2px
    style GCN2 fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    style Output fill:#FFCDD2,stroke:#D32F2F,stroke-width:2px
```

---

## Data Flow: Graph → GNN → Predictions

```mermaid
sequenceDiagram
    participant KG as Knowledge Graph (kg.py)
    participant Neo4j as Neo4j Database
    participant GNN as GNN System
    participant Model as GNN Model
    participant Results as Search Results
    
    Note over KG: Phase 1: Graph Creation
    KG->>Neo4j: Create Case nodes (text, court, date)
    KG->>Neo4j: Create Judge/Court/Statute nodes
    KG->>Neo4j: Create relationships (JUDGED, HEARD_BY)
    KG->>Neo4j: Generate Gemini embeddings (768-dim)
    KG->>Neo4j: Create vector_index + entity_index
    
    Note over Neo4j: Data Storage Complete
    
    Note over GNN: Phase 2: GNN Extraction
    GNN->>Neo4j: MATCH (n) RETURN nodes
    Neo4j-->>GNN: All nodes (Case, Judge, Court, Statute)
    GNN->>Neo4j: MATCH ()-[r]->() RETURN relationships
    Neo4j-->>GNN: All edges (JUDGED, HEARD_BY, REFERENCES)
    
    Note over GNN: Phase 3: Feature Engineering
    GNN->>GNN: Create 9-dim features<br/>(type, court, year)
    GNN->>GNN: Create edge_index tensor
    
    Note over Model: Phase 4: GNN Training
    GNN->>Model: Train on graph structure
    Model->>Model: GCN Layer 1 (9→64)
    Model->>Model: ReLU + Dropout
    Model->>Model: GCN Layer 2 (64→64)
    Model->>Model: Link Prediction Head
    Model-->>GNN: Trained model (ROC-AUC: 0.78-0.85)
    
    Note over GNN: Phase 5: Make Predictions
    GNN->>Model: Predict missing links
    Model-->>GNN: Predicted relationships<br/>(with probabilities)
    GNN->>Neo4j: (Optional) Add predictions to graph
    
    Note over Results: Phase 6: Integrated Search
    Results->>Neo4j: Vector search (Gemini 768-dim)
    Results->>Neo4j: Graph query (relationships)
    Results->>Model: GNN enhancement (predictions)
    Neo4j-->>Results: Similar cases
    Model-->>Results: Related cases
    Results-->>Results: Combine all results
```

---

## Vector Index vs GNN Comparison

```mermaid
graph LR
    subgraph VectorIndex["Vector Index System"]
        V1[Gemini API<br/>768-dim embeddings]
        V2[Neo4j Vector Index<br/>HNSW algorithm]
        V3[Semantic Search<br/>Similar meaning]
        V1 --> V2 --> V3
    end
    
    subgraph GNNSystem["GNN System"]
        G1[Legal Features<br/>9-dim vectors]
        G2[GCN Architecture<br/>Graph learning]
        G3[Link Prediction<br/>Missing relationships]
        G1 --> G2 --> G3
    end
    
    subgraph Neo4jDB["Neo4j Database"]
        N1[Case Nodes]
        N2[Relationships]
        N3[Embeddings]
    end
    
    Neo4jDB --> VectorIndex
    Neo4jDB --> GNNSystem
    
    VectorIndex --> Combined[Combined Search Results]
    GNNSystem --> Combined
    
    style VectorIndex fill:#E3F2FD,stroke:#1976D2,stroke-width:3px
    style GNNSystem fill:#FFF3E0,stroke:#F57C00,stroke-width:3px
    style Neo4jDB fill:#E8F5E9,stroke:#388E3C,stroke-width:3px
    style Combined fill:#F3E5F5,stroke:#7B1FA2,stroke-width:3px
```

---

## Feature Engineering Detail

```mermaid
graph TD
    Node[Legal Node in Graph] --> Check{Node Type?}
    
    Check -->|Case| F1[Features 0-3: 1,0,0,0]
    Check -->|Judge| F2[Features 0-3: 0,1,0,0]
    Check -->|Court| F3[Features 0-3: 0,0,1,0]
    Check -->|Statute| F4[Features 0-3: 0,0,0,1]
    
    F1 --> Title[Feature 4: Title Length / 100]
    F2 --> Title
    F3 --> Title
    F4 --> Title
    
    Title --> CourtCheck{Court Type?}
    
    CourtCheck -->|Supreme Court| C1[Features 5-7: 1,0,0]
    CourtCheck -->|High Court| C2[Features 5-7: 0,1,0]
    CourtCheck -->|District/Other| C3[Features 5-7: 0,0,1]
    
    C1 --> Year[Feature 8: Year - 1950 / 70]
    C2 --> Year
    C3 --> Year
    
    Year --> Final[9-Dimensional Feature Vector<br/>Ready for GNN]
    
    style Node fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    style Final fill:#C8E6C9,stroke:#388E3C,stroke-width:3px
```

---

## GNN Prediction Process

```mermaid
stateDiagram-v2
    [*] --> ExtractGraph: Start GNN Process
    
    state "Extract from Neo4j" as ExtractGraph {
        [*] --> GetNodes: MATCH (n) RETURN nodes
        GetNodes --> GetEdges: MATCH ()-[r]->() RETURN edges
        GetEdges --> [*]: Return graph data
    }
    
    ExtractGraph --> CreateFeatures: Convert to tensors
    
    state "Feature Engineering" as CreateFeatures {
        [*] --> NodeType: Encode node type (4 dims)
        NodeType --> TitleLen: Add title length (1 dim)
        TitleLen --> CourtHier: Add court hierarchy (3 dims)
        CourtHier --> YearInfo: Add year (1 dim)
        YearInfo --> [*]: 9-dim vector ready
    }
    
    CreateFeatures --> TrainGNN: Feed to GNN
    
    state "GNN Training" as TrainGNN {
        [*] --> GCN1: GCN Layer 1 (9→64)
        GCN1 --> ReLU: ReLU + Dropout 0.5
        ReLU --> GCN2: GCN Layer 2 (64→64)
        GCN2 --> LinkPred: Link Prediction Head
        LinkPred --> Sigmoid: Sigmoid (0-1 prob)
        Sigmoid --> [*]: Trained model
    }
    
    TrainGNN --> MakePredictions: Predict links
    
    state "Generate Predictions" as MakePredictions {
        [*] --> Sample: Sample candidate node pairs
        Sample --> Predict: Run through trained model
        Predict --> Rank: Rank by probability
        Rank --> TopK: Select top-k predictions
        TopK --> [*]: Return predictions
    }
    
    MakePredictions --> [*]: Output: Predicted relationships
```

---

## Parallel Systems: Vector Index + GNN

```mermaid
flowchart LR
    subgraph Input["Data Source"]
        KG[Knowledge Graph<br/>in Neo4j]
    end
    
    subgraph VectorSystem["Vector Index Path"]
        V1[Extract Case Text]
        V2[Gemini Embeddings<br/>768 dimensions]
        V3[Vector Index<br/>HNSW algorithm]
        V4[Semantic Search<br/>Similar meaning]
    end
    
    subgraph GNNSystem["GNN Path"]
        G1[Extract Graph Structure<br/>Nodes + Edges]
        G2[Legal Features<br/>9 dimensions]
        G3[GCN Training<br/>Link prediction]
        G4[Relationship Prediction<br/>Missing links]
    end
    
    subgraph Output["Combined Output"]
        R1[Vector Search Results]
        R2[GNN Predicted Links]
        R3[Merged Results]
    end
    
    KG --> V1
    KG --> G1
    
    V1 --> V2 --> V3 --> V4 --> R1
    G1 --> G2 --> G3 --> G4 --> R2
    
    R1 --> R3
    R2 --> R3
    
    style Input fill:#E8F5E9,stroke:#2E7D32,stroke-width:3px
    style VectorSystem fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    style GNNSystem fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    style Output fill:#F3E5F5,stroke:#7B1FA2,stroke-width:3px
```

---

## Search Integration Flow

```mermaid
flowchart TD
    Query[User Query:<br/>Section 65B electronic evidence]
    
    Query --> Router{Query Router}
    
    Router --> Method1[Method 1:<br/>Vector Search]
    Router --> Method2[Method 2:<br/>Graph Query]
    Router --> Method3[Method 3:<br/>GNN Enhancement]
    
    Method1 --> V1[Use Gemini embeddings<br/>768-dim]
    V1 --> V2[Search vector_index<br/>Cosine similarity]
    V2 --> V3[Returns: Anvar P.V. case<br/>89% match]
    
    Method2 --> G1[Generate Cypher query<br/>Graph traversal]
    G1 --> G2[Search relationships<br/>CITES, REFERENCES]
    G2 --> G3[Returns: Connected cases]
    
    Method3 --> GNN1[Use GNN predictions<br/>Predicted links]
    GNN1 --> GNN2[Find cases via predicted<br/>CITES relationships]
    GNN2 --> GNN3[Returns: Related cases<br/>from same court/time]
    
    V3 --> Ranking[Ranking & Scoring]
    G3 --> Ranking
    GNN3 --> Ranking
    
    Ranking --> Final[Final Results:<br/>Top 3-5 most relevant cases]
    
    Final --> Display[Display with:<br/>- Similarity scores<br/>- Case metadata<br/>- LLM analysis]
    
    style Query fill:#6C5CE7,color:#fff,stroke:#fff,stroke-width:2px
    style Ranking fill:#A29BFE,stroke:#6C5CE7,stroke-width:2px
    style Final fill:#00B894,color:#fff,stroke:#fff,stroke-width:3px
    style Display fill:#00CEC9,color:#fff,stroke:#fff,stroke-width:2px
```

---

## How GNN Enhances Vector Search

```mermaid
graph TB
    Start[User searches:<br/>Section 65B evidence] --> VectorSearch[Vector Index finds:<br/>Anvar P.V. case 89%]
    
    VectorSearch --> GNNCheck{GNN trained?}
    
    GNNCheck -->|Yes| GNNQuery[Query GNN for predictions:<br/>What cases cite Anvar P.V.?]
    GNNCheck -->|No| DirectResult[Return vector results only]
    
    GNNQuery --> GNNPredict[GNN predictions:<br/>Case_45 -CITES→ Anvar 92%<br/>Case_78 -CITES→ Anvar 87%<br/>Case_123 -CITES→ Anvar 85%]
    
    GNNPredict --> Enrich[Enrich results with:<br/>+ Predicted citing cases<br/>+ Related precedents<br/>+ Court network analysis]
    
    Enrich --> EnrichedResult[Enhanced Results:<br/>Direct match + Related cases]
    
    DirectResult --> FinalMerge[Merge Results]
    EnrichedResult --> FinalMerge
    
    FinalMerge --> UserDisplay[Display to User]
    
    style Start fill:#6C5CE7,color:#fff,stroke:#fff,stroke-width:2px
    style GNNPredict fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    style EnrichedResult fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    style UserDisplay fill:#00B894,color:#fff,stroke:#fff,stroke-width:3px
```

---

## Usage Instructions

### To view these diagrams:

1. **Copy the Mermaid code** from any section above

2. **Paste into one of these tools:**
   - GitHub/GitLab markdown (renders automatically)
   - [Mermaid Live Editor](https://mermaid.live)
   - VS Code with Mermaid extension
   - Notion (supports Mermaid)

3. **Or include in markdown files:**
   ```markdown
   ```mermaid
   [paste code here]
   ```
   ```

---

## All Diagrams Included:

1. ✅ **Complete System Integration** - Full 5-phase flow
2. ✅ **Simplified GNN Architecture** - Layer-by-layer breakdown
3. ✅ **Data Flow Sequence** - Step-by-step process
4. ✅ **Parallel Systems** - Vector Index vs GNN
5. ✅ **Search Integration** - Query routing and ranking
6. ✅ **GNN Enhancement** - How GNN improves vector search

---

**Copy any section and paste into GitHub README or Mermaid Live Editor!** 🚀

