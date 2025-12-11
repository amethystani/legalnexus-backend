# System Architecture & Sequence Diagrams

## 1. System Architecture Flowchart

```mermaid
graph TD
    subgraph Input
        A[Case Text] --> B[Preprocessing]
        B --> C{Agent Swarm}
    end

    subgraph "Multi-Agent Game (Nash Equilibrium)"
        C --> D[Linker Agent]
        C --> E[Interpreter Agent]
        C --> F[Conflict Agent]
        
        D -- "Propose Citations" --> G((Shared Graph State))
        E -- "Classify Edges" --> G
        F -- "Detect Conflicts" --> G
        
        G -- "Payoff Feedback" --> D
        G -- "Payoff Feedback" --> E
        G -- "Payoff Feedback" --> F
        
        G --> H{Converged?}
        H -- No --> D
        H -- Yes --> I[Final Knowledge Graph]
    end

    subgraph "Hyperbolic GNN"
        I --> J[Hyperbolic Embeddings]
        J --> K[Poincaré Ball Projection]
        K --> L[Link Prediction]
        K --> M[Case Retrieval]
    end

    style C fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
    style I fill:#bfb,stroke:#333,stroke-width:2px
```

## 2. Nash Equilibrium Sequence Diagram

```mermaid
sequenceDiagram
    participant L as Linker Agent
    participant I as Interpreter Agent
    participant C as Conflict Agent
    participant G as Graph State
    participant P as Payoff Function

    Note over L,C: Phase 1: Initial Proposals
    L->>G: Propose Citations (Strategy S_L)
    I->>G: Classify Edges (Strategy S_I)
    
    loop Until Convergence (Nash Equilibrium)
        C->>G: Check for Cycles/Contradictions
        G->>P: Calculate Joint State
        P->>L: Return Payoff U_L (Precision)
        P->>I: Return Payoff U_I (Consistency)
        P->>C: Return Payoff U_C (Logic)
        
        Note right of P: Agents update strategies to maximize payoff
        
        alt Payoff Increases
            L->>G: Refine Citations (Best Response)
            I->>G: Reclassify Edges (Best Response)
            C->>G: Resolve Conflicts (Best Response)
        else Payoff Stable
            Note over L,C: Equilibrium Reached
        end
    end
    
    G->>G: Finalize Knowledge Graph
```

## 3. Hyperbolic Embedding Process

```mermaid
graph LR
    A[Legal Case Node] --> B[Feature Extraction]
    B --> C[GNN Layer 1]
    C --> D[ReLU Activation]
    D --> E[GNN Layer 2]
    E --> F[Exponential Map (Exp_x)]
    F --> G((Poincaré Ball))
    
    subgraph "Euclidean Space"
    B
    C
    D
    E
    end
    
    subgraph "Hyperbolic Space"
    G
    end
    
    style G fill:#ff9,stroke:#333,stroke-width:2px
```
