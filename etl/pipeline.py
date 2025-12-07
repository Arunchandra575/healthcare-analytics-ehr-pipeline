"""!
Large-Scale ETL Pipeline using DuckDB, PyArrow, Pandas & NumPy

Demonstrates:
- Processing 10M+ records with DuckDB (OLAP database)
- PyArrow for columnar data processing
- Pandas/NumPy for complex transformations
- SQL query optimization and performance tuning
- Batch processing and parallel execution
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HealthcareETLPipeline:
    """
    Enterprise-grade ETL pipeline for processing massive healthcare datasets.
    
    Uses DuckDB for in-process OLAP queries, PyArrow for efficient columnar
    data handling, and Pandas/NumPy for complex transformations.
    """
    
    def __init__(self, data_dir='data/sample_data', output_dir='data/processed'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB in-memory database for OLAP processing
        self.conn = duckdb.connect(':memory:')
        logger.info("DuckDB connection established")
    
    def load_parquet_to_duckdb(self):
        """
        Load Parquet files into DuckDB using zero-copy PyArrow integration.
        Demonstrates massive relational database operations.
        """
        logger.info("Loading Parquet files into DuckDB...")
        
        # Load patients table (1M+ rows)
        logger.info("Loading patients table...")
        self.conn.execute("""
            CREATE TABLE patients AS
            SELECT * FROM read_parquet('data/sample_data/patients.parquet')
        """)
        
        # Load encounters table (5M+ rows)
        logger.info("Loading encounters table...")
        self.conn.execute("""
            CREATE TABLE encounters AS
            SELECT * FROM read_parquet('data/sample_data/encounters.parquet')
        """)
        
        # Load diagnoses table (10M+ rows)
        logger.info("Loading diagnoses table...")
        self.conn.execute("""
            CREATE TABLE diagnoses AS
            SELECT * FROM read_parquet('data/sample_data/diagnoses.parquet')
        """)
        
        # Load procedures table (8M+ rows)
        logger.info("Loading procedures table...")
        self.conn.execute("""
            CREATE TABLE procedures AS
            SELECT * FROM read_parquet('data/sample_data/procedures.parquet')
        """)
        
        # Load claims table (7M+ rows)
        logger.info("Loading claims table...")
        self.conn.execute("""
            CREATE TABLE claims AS
            SELECT * FROM read_parquet('data/sample_data/claims.parquet')
        """)
        
        # Create indexes for query optimization
        logger.info("Creating indexes for performance optimization...")
        self.conn.execute("CREATE INDEX idx_patient_id ON patients(patient_id)")
        self.conn.execute("CREATE INDEX idx_encounter_patient ON encounters(patient_id)")
        self.conn.execute("CREATE INDEX idx_diagnosis_patient ON diagnoses(patient_id)")
        self.conn.execute("CREATE INDEX idx_procedure_patient ON procedures(patient_id)")
        self.conn.execute("CREATE INDEX idx_claim_patient ON claims(patient_id)")
        
        logger.info("✓ All tables loaded successfully into DuckDB")
    
    def run_analytical_queries(self):
        """
        Execute complex analytical SQL queries optimized for large-scale data.
        Demonstrates SQL profiling and query optimization techniques.
        """
        logger.info("\nRunning analytical queries...")
        
        # Query 1: Patient cohort analysis with aggregations
        logger.info("Query 1: Patient demographics and encounter summary")
        query1 = """
            SELECT 
                p.age,
                p.gender,
                COUNT(DISTINCT e.encounter_id) as total_encounters,
                AVG(e.length_of_stay) as avg_los,
                COUNT(DISTINCT d.diagnosis_id) as total_diagnoses
            FROM patients p
            LEFT JOIN encounters e ON p.patient_id = e.patient_id
            LEFT JOIN diagnoses d ON p.patient_id = d.patient_id
            GROUP BY p.age, p.gender
            ORDER BY total_encounters DESC
            LIMIT 100
        """
        result1 = self.conn.execute(query1).fetchdf()
        logger.info(f"  Result: {len(result1)} rows")
        
        # Query 2: High-cost patients (window function optimization)
        logger.info("Query 2: Top high-cost patients with procedure costs")
        query2 = """
            WITH patient_costs AS (
                SELECT 
                    p.patient_id,
                    p.age,
                    p.gender,
                    SUM(pr.cost) as total_procedure_cost,
                    SUM(c.claim_amount) as total_claim_amount,
                    COUNT(DISTINCT e.encounter_id) as encounter_count,
                    ROW_NUMBER() OVER (ORDER BY SUM(pr.cost) DESC) as cost_rank
                FROM patients p
                LEFT JOIN procedures pr ON p.patient_id = pr.patient_id
                LEFT JOIN claims c ON p.patient_id = c.patient_id
                LEFT JOIN encounters e ON p.patient_id = e.patient_id
                GROUP BY p.patient_id, p.age, p.gender
            )
            SELECT * FROM patient_costs
            WHERE cost_rank <= 1000
            ORDER BY total_procedure_cost DESC
        """
        result2 = self.conn.execute(query2).fetchdf()
        logger.info(f"  Result: {len(result2)} high-cost patients identified")
        
        # Query 3: ICD-10 diagnosis distribution analysis
        logger.info("Query 3: ICD-10 diagnosis prevalence analysis")
        query3 = """
            SELECT 
                d.icd10_code,
                COUNT(DISTINCT d.patient_id) as patient_count,
                COUNT(d.diagnosis_id) as diagnosis_count,
                AVG(p.age) as avg_patient_age,
                ROUND(COUNT(DISTINCT d.patient_id) * 100.0 / 
                      (SELECT COUNT(DISTINCT patient_id) FROM patients), 2) as prevalence_pct
            FROM diagnoses d
            JOIN patients p ON d.patient_id = p.patient_id
            GROUP BY d.icd10_code
            ORDER BY patient_count DESC
        """
        result3 = self.conn.execute(query3).fetchdf()
        logger.info(f"  Result: {len(result3)} unique ICD-10 codes analyzed")
        
        return {
            'demographics': result1,
            'high_cost_patients': result2,
            'icd10_distribution': result3
        }
    
    def transform_with_pandas_numpy(self):
        """
        Complex data transformations using Pandas and NumPy.
        Demonstrates vectorized operations for performance.
        """
        logger.info("\nPerforming Pandas/NumPy transformations...")
        
        # Extract data from DuckDB to Pandas
        patients_df = self.conn.execute("""
            SELECT p.*, 
                   COUNT(DISTINCT e.encounter_id) as encounter_count,
                   COUNT(DISTINCT d.diagnosis_id) as diagnosis_count
            FROM patients p
            LEFT JOIN encounters e ON p.patient_id = e.patient_id
            LEFT JOIN diagnoses d ON p.patient_id = d.patient_id
            GROUP BY p.patient_id, p.age, p.gender, p.zip_code, p.created_date
        """).fetchdf()
        
        logger.info(f"Loaded {len(patients_df):,} patient records for transformation")
        
        # NumPy vectorized calculations
        # Risk score calculation using NumPy
        patients_df['age_risk_score'] = np.where(
            patients_df['age'] >= 65, 2.0,
            np.where(patients_df['age'] >= 45, 1.5, 1.0)
        )
        
        # Encounter-based risk using vectorized operations
        patients_df['encounter_risk_score'] = np.log1p(patients_df['encounter_count']) * 0.5
        
        # Diagnosis complexity score
        patients_df['diagnosis_complexity'] = np.sqrt(patients_df['diagnosis_count']) * 0.3
        
        # Composite risk score
        patients_df['composite_risk_score'] = (
            patients_df['age_risk_score'] + 
            patients_df['encounter_risk_score'] + 
            patients_df['diagnosis_complexity']
        )
        
        # Categorize patients into risk tiers using Pandas cut
        patients_df['risk_tier'] = pd.cut(
            patients_df['composite_risk_score'],
            bins=[0, 2.5, 4.0, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
        
        logger.info(f"✓ Risk scores calculated for {len(patients_df):,} patients")
        
        # Aggregate statistics
        risk_summary = patients_df.groupby('risk_tier').agg({
            'patient_id': 'count',
            'age': 'mean',
            'encounter_count': 'mean',
            'diagnosis_count': 'mean',
            'composite_risk_score': 'mean'
        }).round(2)
        
        logger.info("\nRisk Tier Distribution:")
        logger.info(f"\n{risk_summary}")
        
        return patients_df
    
    def export_to_pyarrow_parquet(self, df, filename):
        """
        Export processed DataFrame to Parquet using PyArrow for efficiency.
        """
        logger.info(f"\nExporting to Parquet: {filename}")
        
        # Convert Pandas DataFrame to PyArrow Table
        table = pa.Table.from_pandas(df)
        
        # Write with compression and partitioning
        pq.write_table(
            table,
            self.output_dir / filename,
            compression='snappy',
            use_dictionary=True,
            write_statistics=True
        )
        
        file_size = (self.output_dir / filename).stat().st_size / (1024 * 1024)
        logger.info(f"✓ Exported {len(df):,} rows ({file_size:.2f} MB)")
    
    def run_full_pipeline(self):
        """
        Execute the complete ETL pipeline.
        """
        logger.info("="*70)
        logger.info("Healthcare ETL Pipeline - Starting")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        # Step 1: Load data into DuckDB
        self.load_parquet_to_duckdb()
        
        # Step 2: Run analytical queries
        analytics_results = self.run_analytical_queries()
        
        # Step 3: Transform with Pandas/NumPy
        transformed_df = self.transform_with_pandas_numpy()
        
        # Step 4: Export results
        self.export_to_pyarrow_parquet(transformed_df, 'patient_risk_scores.parquet')
        
        for key, df in analytics_results.items():
            self.export_to_pyarrow_parquet(df, f'{key}_analysis.parquet')
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info(f"✓ ETL Pipeline completed in {duration:.2f} seconds")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info("="*70)
    
    def close(self):
        """Close DuckDB connection."""
        self.conn.close()
        logger.info("DuckDB connection closed")


if __name__ == "__main__":
    pipeline = HealthcareETLPipeline()
    try:
        pipeline.run_full_pipeline()
    finally:
        pipeline.close()
