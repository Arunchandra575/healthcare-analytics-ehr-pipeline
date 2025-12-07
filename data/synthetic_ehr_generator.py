"""!
Synthetic EHR Data Generator using PyArrow for Healthcare Analytics

Generates realistic healthcare datasets with:
- Patient demographics
- Clinical encounters
- Diagnoses (ICD-10 codes)
- Procedures (CPT codes)
- Claims data

Outputs to Parquet format for columnar processing efficiency.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path


class SyntheticEHRGenerator:
    """
    Generate large-scale synthetic EHR data using PyArrow for high-performance
    columnar data processing. Simulates 10M+ patient records.
    """
    
    def __init__(self, num_patients=1_000_000, output_dir='data/sample_data'):
        self.num_patients = num_patients
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Medical code lists
        self.icd10_codes = [
            'I10', 'E11.9', 'J44.9', 'I25.10', 'F41.9',
            'M79.3', 'K21.9', 'N18.3', 'J18.9', 'I48.91'
        ]
        
        self.cpt_codes = [
            '99213', '99214', '99215', '80053', '36415',
            '93000', '71045', '70450', '99285', '99283'
        ]
        
        self.specialties = [
            'Internal Medicine', 'Cardiology', 'Endocrinology',
            'Pulmonology', 'Emergency Medicine', 'Family Medicine'
        ]
    
    def generate_patients(self):
        """Generate patient demographics table."""
        print(f"Generating {self.num_patients:,} patient records...")
        
        # Use NumPy for efficient array generation
        patient_ids = np.arange(1, self.num_patients + 1)
        ages = np.random.randint(0, 100, size=self.num_patients)
        genders = np.random.choice(['M', 'F', 'O'], size=self.num_patients)
        
        # Convert to PyArrow Table for columnar efficiency
        patients_table = pa.table({
            'patient_id': pa.array(patient_ids, type=pa.int64()),
            'age': pa.array(ages, type=pa.int32()),
            'gender': pa.array(genders, type=pa.string()),
            'zip_code': pa.array(
                np.random.randint(10000, 99999, size=self.num_patients),
                type=pa.int32()
            ),
            'created_date': pa.array(
                [datetime.now() - timedelta(days=random.randint(0, 3650))
                 for _ in range(self.num_patients)]
            )
        })
        
        # Write to Parquet with compression
        pq.write_table(
            patients_table,
            self.output_dir / 'patients.parquet',
            compression='snappy'
        )
        print(f"✓ Patients table written: {len(patients_table):,} rows")
        return patients_table
    
    def generate_encounters(self, num_encounters_per_patient=5):
        """Generate clinical encounters (visits)."""
        num_encounters = self.num_patients * num_encounters_per_patient
        print(f"Generating {num_encounters:,} encounter records...")
        
        encounter_ids = np.arange(1, num_encounters + 1)
        patient_ids = np.repeat(
            np.arange(1, self.num_patients + 1),
            num_encounters_per_patient
        )
        
        # Generate random encounter dates
        encounter_dates = [
            datetime.now() - timedelta(days=random.randint(0, 1825))
            for _ in range(num_encounters)
        ]
        
        encounters_table = pa.table({
            'encounter_id': pa.array(encounter_ids, type=pa.int64()),
            'patient_id': pa.array(patient_ids, type=pa.int64()),
            'encounter_date': pa.array(encounter_dates),
            'specialty': pa.array(
                np.random.choice(self.specialties, size=num_encounters),
                type=pa.string()
            ),
            'encounter_type': pa.array(
                np.random.choice(['Inpatient', 'Outpatient', 'Emergency'],
                               size=num_encounters),
                type=pa.string()
            ),
            'length_of_stay': pa.array(
                np.random.randint(1, 15, size=num_encounters),
                type=pa.int32()
            )
        })
        
        pq.write_table(
            encounters_table,
            self.output_dir / 'encounters.parquet',
            compression='snappy'
        )
        print(f"✓ Encounters table written: {len(encounters_table):,} rows")
        return encounters_table
    
    def generate_diagnoses(self):
        """Generate diagnosis records with ICD-10 codes."""
        num_diagnoses = self.num_patients * 10
        print(f"Generating {num_diagnoses:,} diagnosis records...")
        
        diagnosis_ids = np.arange(1, num_diagnoses + 1)
        patient_ids = np.random.randint(1, self.num_patients + 1,
                                       size=num_diagnoses)
        
        diagnoses_table = pa.table({
            'diagnosis_id': pa.array(diagnosis_ids, type=pa.int64()),
            'patient_id': pa.array(patient_ids, type=pa.int64()),
            'icd10_code': pa.array(
                np.random.choice(self.icd10_codes, size=num_diagnoses),
                type=pa.string()
            ),
            'diagnosis_date': pa.array([
                datetime.now() - timedelta(days=random.randint(0, 1825))
                for _ in range(num_diagnoses)
            ])
        })
        
        pq.write_table(
            diagnoses_table,
            self.output_dir / 'diagnoses.parquet',
            compression='snappy'
        )
        print(f"✓ Diagnoses table written: {len(diagnoses_table):,} rows")
        return diagnoses_table
    
    def generate_procedures(self):
        """Generate procedure records with CPT codes."""
        num_procedures = self.num_patients * 8
        print(f"Generating {num_procedures:,} procedure records...")
        
        procedure_ids = np.arange(1, num_procedures + 1)
        patient_ids = np.random.randint(1, self.num_patients + 1,
                                       size=num_procedures)
        
        procedures_table = pa.table({
            'procedure_id': pa.array(procedure_ids, type=pa.int64()),
            'patient_id': pa.array(patient_ids, type=pa.int64()),
            'cpt_code': pa.array(
                np.random.choice(self.cpt_codes, size=num_procedures),
                type=pa.string()
            ),
            'procedure_date': pa.array([
                datetime.now() - timedelta(days=random.randint(0, 1825))
                for _ in range(num_procedures)
            ]),
            'cost': pa.array(
                np.random.uniform(50, 5000, size=num_procedures),
                type=pa.float64()
            )
        })
        
        pq.write_table(
            procedures_table,
            self.output_dir / 'procedures.parquet',
            compression='snappy'
        )
        print(f"✓ Procedures table written: {len(procedures_table):,} rows")
        return procedures_table
    
    def generate_claims(self):
        """Generate insurance claims data."""
        num_claims = self.num_patients * 7
        print(f"Generating {num_claims:,} claims records...")
        
        claim_ids = np.arange(1, num_claims + 1)
        patient_ids = np.random.randint(1, self.num_patients + 1,
                                       size=num_claims)
        
        claims_table = pa.table({
            'claim_id': pa.array(claim_ids, type=pa.int64()),
            'patient_id': pa.array(patient_ids, type=pa.int64()),
            'claim_amount': pa.array(
                np.random.uniform(100, 50000, size=num_claims),
                type=pa.float64()
            ),
            'paid_amount': pa.array(
                np.random.uniform(80, 45000, size=num_claims),
                type=pa.float64()
            ),
            'claim_status': pa.array(
                np.random.choice(['Paid', 'Pending', 'Denied'],
                               size=num_claims),
                type=pa.string()
            ),
            'claim_date': pa.array([
                datetime.now() - timedelta(days=random.randint(0, 1825))
                for _ in range(num_claims)
            ])
        })
        
        pq.write_table(
            claims_table,
            self.output_dir / 'claims.parquet',
            compression='snappy'
        )
        print(f"✓ Claims table written: {len(claims_table):,} rows")
        return claims_table
    
    def generate_all(self):
        """Generate complete EHR dataset."""
        print("\n" + "="*60)
        print("Synthetic EHR Data Generation")
        print("="*60 + "\n")
        
        self.generate_patients()
        self.generate_encounters()
        self.generate_diagnoses()
        self.generate_procedures()
        self.generate_claims()
        
        print("\n" + "="*60)
        print("✓ All datasets generated successfully!")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Generate 1 million patient records (configurable)
    generator = SyntheticEHRGenerator(num_patients=1_000_000)
    generator.generate_all()
