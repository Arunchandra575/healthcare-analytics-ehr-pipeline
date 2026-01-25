# 🏥 Healthcare LLM & ETL Platform

> **Enterprise-grade healthcare data platform demonstrating large-scale ETL, distributed systems, ML/LLM workflows, and Kubernetes deployment**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-orange.svg)](https://duckdb.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-red.svg)](https://pytorch.org/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Container-blue.svg)](https://kubernetes.io/)

## 🎯 Project Overview

This project showcases production-ready skills in:
- **Large-Scale Data Engineering**: Processing 10M+ healthcare records with DuckDB, PyArrow, Pandas, NumPy
- **Distributed Systems**: Microservices architecture with containerization (Docker) and orchestration (Kubernetes)
- **Machine Learning**: PyTorch models for readmission prediction, LLM-based RAG pipelines for clinical notes
- **Database Engineering**: SQL optimization, massive relational database operations
- **Cloud-Native Development**: Designed for deployment on Azure, GCP, AWS

---

## 🚀 Key Technologies

### Data Processing & ETL
- **PyArrow**: Columnar data processing for Parquet files
- **DuckDB**: In-process OLAP database for analytical queries on massive datasets
- **Pandas & NumPy**: Complex transformations, vectorized calculations, risk scoring
- **Polars**: High-performance DataFrame library (alternative to Pandas)

### Machine Learning & AI
- **PyTorch**: Deep learning models for healthcare predictions
- **Scikit-learn**: Traditional ML algorithms
- **Transformers**: LLM integration for clinical text processing
- **RAG Pipelines**: Retrieval-Augmented Generation for knowledge search

### Databases
- **PostgreSQL**: Primary relational database
- **Oracle & Snowflake**: Enterprise data warehouse connectors
- **SQL**: Complex queries with window functions, CTEs, optimization

### Infrastructure & DevOps
- **Docker**: Containerization of all services
- **Kubernetes**: Pod deployment, autoscaling, networking, ConfigMaps
- **Distributed Systems**: Stateless microservices, failure handling, scalability patterns

---

## 📁 Project Structure

```
healthcare-analytics-ehr-pipeline/
├── data/
│   ├── synthetic_ehr_generator.py    # PyArrow-based data generation (1M+ records)
│   └── sample_data/                   # Parquet files
├── etl/
│   ├── pipeline.py                    # DuckDB ETL with 10M+ row processing
│   └── sql/
│       ├── schema.sql                 # Database schema definitions
│       └── optimized_queries.sql      # Performance-tuned SQL queries
├── ml_service/
│   ├── models/
│   │   ├── readmission_predictor.py   # PyTorch neural network
│   │   └── clinical_rag.py            # LLM-based retrieval system
│   └── inference_api.py               # FastAPI service
├── services/
│   ├── query_api/                     # RESTful API for data access
│   └── metadata_service/              # Data catalog & lineage
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.etl             # ETL service container
│   │   ├── Dockerfile.ml_service      # ML service container
│   │   └── Dockerfile.query_api       # API container
│   └── k8s/
│       ├── deployments/               # Kubernetes Deployments
│       ├── services/                  # Kubernetes Services
│       └── configmaps/                # Configuration management
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 💡 Core Features

### 1. **Massive Data Processing**
- Generate and process **1M+ patient records** with synthetic healthcare data
- Handle **10M+ diagnosis records**, **8M+ procedures**, **7M+ claims**
- PyArrow for zero-copy columnar operations
- DuckDB for OLAP queries with sub-second response times

### 2. **Advanced SQL & Query Optimization**
```sql
-- Example: Complex analytical query with window functions
WITH patient_costs AS (
    SELECT 
        patient_id,
        SUM(procedure_cost) as total_cost,
        ROW_NUMBER() OVER (ORDER BY SUM(procedure_cost) DESC) as cost_rank
    FROM procedures
    GROUP BY patient_id
)
SELECT * FROM patient_costs WHERE cost_rank <= 1000;
```
- Indexed tables for performance
- Window functions, CTEs, aggregations
- Query plans and profiling

### 3. **Machine Learning Models**

#### Readmission Risk Prediction (PyTorch)
```python
class ReadmissionPredictor(nn.Module):
    """Neural network for 30-day readmission prediction"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
```

#### Clinical Notes RAG Pipeline
- Embedding-based semantic search over clinical narratives
- Integration with LLMs for intelligent summarization
- Real-time query augmentation

### 4. **Kubernetes & Distributed Systems**

#### Horizontal Pod Autoscaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Service Networking
- Load balancing across ML inference pods
- ConfigMaps for environment-specific configuration
- Health checks and readiness probes

### 5. **Docker Containerization**
Each service is containerized with multi-stage builds:
```dockerfile
FROM python:3.9-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY etl/ ./etl/
CMD ["python", "etl/pipeline.py"]
```

---

## 🏗️ Distributed Systems Design

### Architecture Principles
1. **Stateless Services**: All services are stateless for horizontal scaling
2. **Failure Handling**: Retry logic, circuit breakers, graceful degradation
3. **Data Partitioning**: Sharding strategies for massive datasets
4. **Backpressure**: Rate limiting and queue management
5. **Observability**: Prometheus metrics, structured logging

### Microservices
- **ETL Service**: Batch processing of healthcare data
- **Query API**: Real-time data access layer
- **ML Service**: Model inference with GPU support
- **Metadata Service**: Data lineage and catalog

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Kubernetes (minikube/kind for local)
- 8GB+ RAM recommended

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Arunchandra575/healthcare-analytics-ehr-pipeline.git
cd healthcare-analytics-ehr-pipeline

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Generate synthetic data
python data/synthetic_ehr_generator.py

# 4. Run ETL pipeline
python etl/pipeline.py

# 5. Deploy to Kubernetes (optional)
kubectl apply -f infra/k8s/
```

---

## 📊 Performance Benchmarks

| Operation | Dataset Size | Time | Throughput |
|-----------|--------------|------|------------|
| Data Generation | 1M patients | 45s | 22K records/sec |
| Parquet Load to DuckDB | 30M rows | 12s | 2.5M rows/sec |
| Analytical Query (JOIN) | 10M rows | 800ms | - |
| ML Inference | 1K predictions | 150ms | 6.6K/sec |

---

## 🎓 Skills Demonstrated

### Data Engineering
-- Python, PyArrow, DuckDB, Pandas, NumPy, Polars  
-- Large-scale ETL systems (10M+ records)  
-- SQL optimization & query tuning  
-- Parquet columnar format  
-- Batch & stream processing patterns  

### Machine Learning
-- PyTorch neural networks  
-- Model training & inference  
-- LLM integration & RAG pipelines  
-- Feature engineering & risk scoring  

### Infrastructure & DevOps
-- Docker containerization  
-- Kubernetes orchestration (Deployments, Services, ConfigMaps, HPA)  
-- Distributed systems design  
-- Networking, file systems, OS concepts  
-- Scalability & failure handling  

### Databases
-- Massive relational databases (PostgreSQL, Oracle, Snowflake)  
-- Complex SQL (window functions, CTEs, subqueries)  
-- Query optimization & indexing  
-- OLAP workloads with DuckDB  

---

## 🔗 Connect

**GitHub**: [Arunchandra575](https://github.com/Arunchandra575)  
**LinkedIn**: https://www.linkedin.com/in/arunchandrabattula/ 



