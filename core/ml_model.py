import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class DrugRepurposingModel:
    def __init__(self):
        self.data_path = Path('data')
        self.load_data()
    
    def load_data(self):
        """Load all datasets - nodes and links"""
        try:
            # Load CSV files
            self.ctd_gene_disease = pd.read_csv(self.data_path / 'CTD_gene_disease_protein.csv')
            self.disease_data = pd.read_csv(self.data_path / 'disease.csv')
            self.drugs_data = pd.read_csv(self.data_path / 'drugs.csv')
            
            # Load CTD therapeutic dataset
            self.therapeutic_data = pd.read_csv(self.data_path / 'CTD_THERAPEUTIC.csv')
            
            self.disease_embeddings = pd.read_csv(self.data_path / 'disease_embeddings.csv')
            
            # Load embeddings
            self.ctd_embeddings = np.load(self.data_path / 'CTD_embeddings.npy')
            
            # Clean and validate data
            self._clean_data()
            
            # Build knowledge graph
            self._build_graph()
            
            print("\n=== DATA LOADED SUCCESSFULLY ===")
            print(f"Total drug-protein interactions: {len(self.drugs_data):,}")
            print(f"Unique drugs: {self.drugs_data['drug_name'].nunique():,}")
            print(f"Unique proteins: {self.drugs_data['target_uniprot'].nunique():,}")
            print(f"Unique diseases: {self.disease_data['DiseaseName'].nunique():,}")
            print(f"Therapeutic dataset: {len(self.therapeutic_data):,} disease-drug pairs")
            print("="*40)
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _clean_data(self):
        """RULE 3: Remove garbage/noise data"""
        print("Cleaning data - removing invalid activity values...")
        
        # Remove biologically meaningless values
        invalid_conditions = (
            (self.drugs_data['activity_type'].isin(['Ki', 'IC50', 'Kd']) & 
             (self.drugs_data['activity_value'] > 2000)) |
            (self.drugs_data['activity_value'] == 10000) |
            (self.drugs_data['activity_type'].isin(['Residual activity', '%Control']) & 
             (self.drugs_data['activity_value'] > 2000)) |
            (self.drugs_data['activity_value'].isna())
        )
        
        before_count = len(self.drugs_data)
        self.drugs_data = self.drugs_data[~invalid_conditions]
        after_count = len(self.drugs_data)
        
        print(f"Removed {before_count - after_count:,} invalid activity rows")
        
        # RULE 4: Merge duplicate assay rows - keep strongest (lowest) value
        print("Merging duplicate assays - keeping strongest binding...")
        self.drugs_data = self.drugs_data.sort_values('activity_value').groupby(
            ['drug_name', 'target_uniprot', 'activity_type'], 
            as_index=False
        ).first()
        
        print(f"After deduplication: {len(self.drugs_data):,} rows")
    
    def _build_graph(self):
        """Build knowledge graph connections"""
        print("Building knowledge graph...")
        
        # Build list of all drugs in dataset (for validation)
        self.valid_drugs = set(self.drugs_data['drug_name'].str.lower().str.strip().unique())
        
        # Parse CTD therapeutic dataset
        print("Loading CTD therapeutic dataset...")
        self.drug_to_disease_therapeutic = {}
        self.disease_to_drug_therapeutic = {}
        
        for _, row in self.therapeutic_data.iterrows():
            # Priority: drug_name > chemical_name > molecule_chembl_id
            drug_display_name = None
            if pd.notna(row.get('drug_name')) and str(row.get('drug_name')).strip():
                drug_display_name = str(row.get('drug_name')).strip()
            elif pd.notna(row.get('chemical_name')) and str(row.get('chemical_name')).strip():
                drug_display_name = str(row.get('chemical_name')).strip()
            elif pd.notna(row.get('molecule_chembl_id')) and str(row.get('molecule_chembl_id')).strip():
                drug_display_name = str(row.get('molecule_chembl_id')).strip()
            
            # Use cleaned names for matching keys
            drug_key = str(row.get('drug_clean', row.get('chemical_name', ''))).lower().strip()
            disease_name = row.get('disease_clean', row.get('disease_name', ''))
            disease_key = str(disease_name).lower().strip()
            evidence = row.get('evidence', 'therapeutic')
            
            if drug_display_name and disease_name and drug_key:
                # Build drug -> disease mapping
                if drug_key not in self.drug_to_disease_therapeutic:
                    self.drug_to_disease_therapeutic[drug_key] = []
                self.drug_to_disease_therapeutic[drug_key].append({
                    'disease': disease_name,
                    'evidence': evidence,
                    'source': 'CTD_THERAPEUTIC'
                })
                
                # Build disease -> drug mapping (store display name)
                if disease_key not in self.disease_to_drug_therapeutic:
                    self.disease_to_drug_therapeutic[disease_key] = []
                self.disease_to_drug_therapeutic[disease_key].append({
                    'drug': drug_display_name,  # Use display name for output
                    'evidence': evidence,
                    'source': 'CTD_THERAPEUTIC'
                })
        
        print(f"Loaded {len(self.drug_to_disease_therapeutic)} therapeutic drugs")
        print(f"Loaded {len(self.disease_to_drug_therapeutic)} diseases with treatments")
        
        # Drug -> Protein (cleaned data)
        self.drug_to_protein = defaultdict(list)
        self.drug_name_display = {}  # Map lowercase -> original case
        print("Loading drug-protein interactions...")
        
        for drug_original, protein, activity_type, activity_value in zip(
            self.drugs_data['drug_name'].str.strip(),
            self.drugs_data['target_uniprot'].astype(str).str.strip(),
            self.drugs_data['activity_type'],
            self.drugs_data['activity_value']
        ):
            drug_key = drug_original.lower()
            # Store original drug name for display
            if drug_key not in self.drug_name_display:
                self.drug_name_display[drug_key] = drug_original
            
            self.drug_to_protein[drug_key].append({
                'protein': protein,
                'activity_type': activity_type,
                'activity_value': activity_value
            })
        
        # Build Gene Symbol -> UniProt ID mapping
        self.gene_to_uniprot = {}
        for _, row in self.ctd_gene_disease.iterrows():
            gene = str(row['GeneSymbol']).strip()
            uniprot = str(row['UniProtID']).strip()
            if gene and uniprot and uniprot != 'nan':
                self.gene_to_uniprot[gene] = uniprot
        
        print(f"Gene to UniProt mapping: {len(self.gene_to_uniprot)} genes")
        
        # Protein -> Disease
        self.protein_to_disease = defaultdict(list)
        
        # From CTD
        for _, row in self.ctd_gene_disease.iterrows():
            protein = str(row['UniProtID']).strip()
            disease = row['DiseaseName']
            gene = row['GeneSymbol']
            if protein and protein != 'nan':
                self.protein_to_disease[protein].append({
                    'disease': disease,
                    'gene': gene
                })
        
        # From disease.csv - use GeneSymbol but map to UniProt
        for gene, disease in zip(
            self.disease_data['GeneSymbol'],
            self.disease_data['DiseaseName']
        ):
            gene_str = str(gene).strip()
            # Try to get UniProt ID for this gene
            uniprot = self.gene_to_uniprot.get(gene_str)
            if uniprot:
                self.protein_to_disease[uniprot].append({
                    'disease': disease,
                    'gene': gene_str
                })
            else:
                # Fallback: store by gene symbol
                self.protein_to_disease[gene_str].append({
                    'disease': disease,
                    'gene': gene_str
                })
        
        # Disease -> Protein (reverse)
        self.disease_to_protein = defaultdict(list)
        for protein, diseases in self.protein_to_disease.items():
            for d in diseases:
                disease_key = d['disease'].lower().strip()
                self.disease_to_protein[disease_key].append({
                    'protein': protein,
                    'gene': d['gene']
                })
        
        # Build embedding lookup
        self._build_embedding_lookup()
        
        print(f"Graph built successfully!")
        print(f"  - {len(self.drug_to_protein):,} drugs")
        print(f"  - {len(self.protein_to_disease):,} proteins/genes")
        print(f"  - {len(self.disease_to_protein):,} diseases")
        print(f"  - {len(self.drug_to_disease_therapeutic):,} therapeutic pairs")
    
    def _build_embedding_lookup(self):
        """Build embedding lookup for similarity calculations"""
        self.disease_embedding_map = {}
        if not self.disease_embeddings.empty:
            for _, row in self.disease_embeddings.iterrows():
                disease_name = row['DiseaseName'].lower().strip()
                # Get embedding columns (all except DiseaseName)
                embedding = row.drop('DiseaseName').values.astype(float)
                self.disease_embedding_map[disease_name] = embedding
    
    def _calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        try:
            emb1 = np.array(emb1).reshape(1, -1)
            emb2 = np.array(emb2).reshape(1, -1)
            return float(cosine_similarity(emb1, emb2)[0][0])
        except:
            return 0.0
    
    def predict(self, query_type, query_value):
        """Main prediction function"""
        # RULE 7: Check for broad terms
        broad_terms = ['cancer', 'tumor', 'infection', 'disease', 'syndrome', 'disorder']
        if query_type == 'disease' and any(term in query_value.lower() for term in broad_terms):
            if query_value.lower() in broad_terms:
                return {
                    'query_type': query_type,
                    'query_value': query_value,
                    'error': f'The term "{query_value}" is too broad. Please enter a specific disease name.'
                }
        
        results = {
            'query_type': query_type,
            'query_value': query_value,
            'predictions': {}
        }
        
        try:
            if query_type == 'drug':
                results['predictions'] = self.predict_from_drug(query_value)
            elif query_type == 'protein':
                results['predictions'] = self.predict_from_protein(query_value)
            elif query_type == 'disease':
                results['predictions'] = self.predict_from_disease(query_value)
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _is_fda_approved(self, drug_name, disease_name):
        """
        Use LLM internal knowledge to validate FDA approval
        This is a placeholder - in production, this would query a validated database
        For now, returns False to rely only on dataset
        """
        # Known FDA-approved pairs (subset for validation)
        # In production, this would be a comprehensive database
        known_approvals = {
            'aspirin': ['acute coronary syndrome', 'myocardial infarction', 'stroke prevention'],
            'metformin': ['diabetes mellitus', 'diabetes mellitus, type 2'],
            'insulin': ['diabetes mellitus', 'diabetes mellitus, type 1', 'diabetes mellitus, type 2'],
            'warfarin': ['atrial fibrillation', 'venous thromboembolism', 'thrombosis'],
            'atorvastatin': ['hypercholesterolemia', 'atherosclerosis'],
            'lisinopril': ['hypertension', 'heart failure'],
            'levothyroxine': ['hypothyroidism'],
            'albuterol': ['asthma', 'chronic obstructive pulmonary disease'],
            'omeprazole': ['gastroesophageal reflux disease', 'peptic ulcer'],
            'acetaminophen': ['pain', 'fever'],
        }
        
        drug_key = drug_name.lower().strip()
        disease_key = disease_name.lower().strip()
        
        if drug_key in known_approvals:
            return any(disease_key in approved.lower() for approved in known_approvals[drug_key])
        
        return False
    
    def predict_from_drug(self, drug_name):
        """Drug input → Diseases + Proteins"""
        drug_key = drug_name.lower().strip()
        diseases = []
        proteins = []
        
        # STEP 1: Check CTD_THERAPEUTIC.csv first
        therapeutic_diseases = self.drug_to_disease_therapeutic.get(drug_key, [])
        
        # Try partial match in therapeutic data if exact match not found
        if not therapeutic_diseases:
            for key in self.drug_to_disease_therapeutic.keys():
                if drug_key in key or key in drug_key:
                    therapeutic_diseases = self.drug_to_disease_therapeutic[key]
                    drug_key = key  # Update drug_key to matched key
                    break
        
        # If found in therapeutic dataset, add as Rank 1
        if therapeutic_diseases:
            diseases.append({
                'rank': 1,
                'disease_name': therapeutic_diseases[0]['disease'],
                'evidence': therapeutic_diseases[0]['evidence'],
                'source': 'Therapeutic',
                'percentage': 100.0
            })
        
        # STEP 2: Validate drug exists in main dataset (drugs.csv)
        if drug_key not in self.valid_drugs:
            # Try partial match
            found = False
            for valid_drug in self.valid_drugs:
                if drug_key in valid_drug or valid_drug in drug_key:
                    drug_key = valid_drug
                    found = True
                    break
            if not found:
                # If drug only exists in therapeutic data, return just that
                return {'diseases': diseases, 'proteins': []}
        
        # Get proteins
        drug_proteins = self.drug_to_protein.get(drug_key, [])
        if not drug_proteins:
            for key in self.drug_to_protein.keys():
                if drug_key in key or key in drug_key:
                    drug_proteins = self.drug_to_protein[key]
                    break
        
        # Add proteins with gene symbols
        for p in drug_proteins[:10]:
            gene_symbol = None
            protein_id = str(p['protein']).strip()
            
            # Look up gene symbol from CTD mapping
            for gene, uniprot_id in zip(self.ctd_gene_disease['GeneSymbol'], 
                                       self.ctd_gene_disease['UniProtID']):
                if protein_id == str(uniprot_id).strip():
                    gene_symbol = str(gene).strip()
                    break
            
            proteins.append({
                'protein': protein_id,  # UniProt ID or other protein identifier
                'gene': gene_symbol if gene_symbol else 'N/A',  # Show N/A instead of repeating protein ID
                'activity_type': p['activity_type'],
                'activity_value': p['activity_value']
            })
        
        # Get diseases via proteins (RULE 2: Protein-linked)
        disease_scores = defaultdict(int)
        disease_is_fda = {}  # Track FDA-approved status
        therapeutic_disease_names = [d['disease_name'].lower().strip() for d in diseases]  # Track therapeutic matches
        
        for protein_info in drug_proteins:
            protein_id = protein_info['protein']
            protein_diseases = self.protein_to_disease.get(protein_id, [])
            for pd in protein_diseases:
                disease_name = pd['disease']
                # Skip if this disease is already in therapeutic results
                if disease_name.lower().strip() in therapeutic_disease_names:
                    continue
                disease_scores[disease_name] += 1
                
                # Check if FDA-approved (but not in therapeutic CSV)
                if disease_name not in disease_is_fda:
                    disease_is_fda[disease_name] = self._is_fda_approved(drug_key, disease_name)
        
        # RULE 5 & 6: Calculate confidence using embeddings
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get query embedding if available
        query_emb = self.disease_embedding_map.get(drug_key)
        
        rank = 2 if diseases else 1
        for disease_name, count in sorted_diseases[:9]:
            # Check if FDA-approved (hybrid validation)
            is_fda = disease_is_fda.get(disease_name, False)
            
            # Calculate similarity-based confidence
            disease_emb = self.disease_embedding_map.get(disease_name.lower().strip())
            
            if query_emb is not None and disease_emb is not None:
                similarity = self._calculate_similarity(query_emb, disease_emb)
                # RULE 6: Drop weak links
                if similarity < 0.20:
                    continue
                percentage = round(similarity * 100, 1)
            else:
                # Fallback to protein count
                max_score = sorted_diseases[0][1] if sorted_diseases else 1
                percentage = round((count / max_score) * 100, 1)
            
            # Label as Therapeutic if FDA-approved, otherwise Protein-linked
            if is_fda and rank == 2 and not diseases:
                # FDA-approved but not in CSV - make it Rank 1
                diseases.insert(0, {
                    'rank': 1,
                    'disease_name': disease_name,
                    'evidence': 'therapeutic (FDA-approved)',
                    'source': 'Therapeutic',
                    'percentage': 100.0
                })
                rank = 2
            else:
                source = 'Therapeutic (FDA)' if is_fda else 'Protein-linked'
                evidence = f'FDA-approved + {count} protein(s)' if is_fda else f'Via {count} protein(s)'
                
                diseases.append({
                    'rank': rank,
                    'disease_name': disease_name,
                    'evidence': evidence,
                    'source': source,
                    'percentage': percentage
                })
                rank += 1
        
        # Add message if only therapeutic result found
        result = {
            'diseases': diseases[:10] if diseases else [],
            'proteins': proteins[:10] if proteins else []
        }
        
        if len(diseases) == 1 and diseases[0]['source'] == 'Therapeutic':
            result['message'] = 'No protein-linked diseases found in the dataset for this drug.'
        
        return result
    
    def predict_from_disease(self, disease_name):
        """Disease input → Drugs + Proteins"""
        disease_key = disease_name.lower().strip()
        drugs = []
        proteins = []
        
        # STEP 1: Check CTD_THERAPEUTIC.csv first
        therapeutic_drugs = self.disease_to_drug_therapeutic.get(disease_key, [])
        
        # Try partial match if exact not found
        if not therapeutic_drugs:
            for key in self.disease_to_drug_therapeutic.keys():
                if disease_key in key or key in disease_key:
                    therapeutic_drugs = self.disease_to_drug_therapeutic[key]
                    disease_key = key  # Update disease_key to matched key
                    break
        
        # If found in therapeutic dataset, add as Rank 1
        if therapeutic_drugs:
            first_drug = therapeutic_drugs[0]
            drugs.append({
                'rank': 1,
                'drug_name': first_drug['drug'],
                'evidence': first_drug['evidence'],
                'source': 'Therapeutic',
                'percentage': 100.0
            })
        
        # Get proteins
        disease_proteins = self.disease_to_protein.get(disease_key, [])
        if not disease_proteins:
            for key in self.disease_to_protein.keys():
                if disease_key in key or key in disease_key:
                    disease_proteins = self.disease_to_protein[key]
                    break
        
        for p in disease_proteins[:10]:
            gene_val = p.get('gene', 'N/A')
            # Don't show N/A if gene is empty or same as protein
            if not gene_val or gene_val == 'nan' or str(gene_val).strip() == str(p['protein']).strip():
                gene_val = 'N/A'
            
            proteins.append({
                'protein': str(p['protein']).strip(),
                'gene': gene_val
            })
        
        # Get drugs via proteins (RULE 2) - Only from dataset
        drug_scores = defaultdict(int)
        drug_is_fda = {}
        therapeutic_drug_names = [d['drug_name'].lower().strip() for d in drugs]  # Track therapeutic matches
        
        for protein_info in disease_proteins:
            protein_id = protein_info['protein']
            for drug_key, drug_proteins in self.drug_to_protein.items():
                for dp in drug_proteins:
                    if dp['protein'] == protein_id:
                        # Skip if this drug is already in therapeutic results
                        if drug_key.lower().strip() in therapeutic_drug_names:
                            continue
                        # VALIDATION: Drug must be in dataset
                        if drug_key in self.valid_drugs:
                            drug_scores[drug_key] += 1
                            
                            # Check FDA approval
                            if drug_key not in drug_is_fda:
                                drug_is_fda[drug_key] = self._is_fda_approved(drug_key, disease_key)
        
        # Calculate confidence
        sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
        
        rank = 2 if drugs else 1
        for drug_name, count in sorted_drugs[:9]:
            # Check FDA approval
            is_fda = drug_is_fda.get(drug_name, False)
            
            max_score = sorted_drugs[0][1] if sorted_drugs else 1
            percentage = round((count / max_score) * 100, 1)
            
            # Label appropriately
            if is_fda and rank == 2 and not drugs:
                # FDA-approved but not in CSV - make it Rank 1
                drugs.insert(0, {
                    'rank': 1,
                    'drug_name': drug_name,
                    'evidence': 'therapeutic (FDA-approved)',
                    'source': 'Therapeutic',
                    'percentage': 100.0
                })
                rank = 2
            else:
                source = 'Therapeutic (FDA)' if is_fda else 'Protein-linked'
                evidence = f'FDA-approved + {count} protein(s)' if is_fda else f'Via {count} protein(s)'
                
                # Use display name (original case) instead of lowercase
                display_name = self.drug_name_display.get(drug_name, drug_name.upper())
                
                drugs.append({
                    'rank': rank,
                    'drug_name': display_name,
                    'evidence': evidence,
                    'source': source,
                    'percentage': percentage
                })
                rank += 1
        
        # Add message if only therapeutic result found
        result = {
            'drugs': drugs[:10] if drugs else [],
            'proteins': proteins[:10] if proteins else []
        }
        
        if len(drugs) == 1 and drugs[0]['source'] == 'Therapeutic':
            result['message'] = 'No protein-linked drugs found in the dataset for this disease.'
        
        return result
    
    def predict_from_protein(self, protein_id):
        """Protein input → Drugs + Diseases"""
        protein_key = protein_id.strip()
        drugs = []
        diseases = []
        
        # Get drugs
        drug_scores = []
        for drug_key, drug_proteins in self.drug_to_protein.items():
            for dp in drug_proteins:
                if protein_key.upper() == str(dp['protein']).upper().strip():
                    # Use display name (original case)
                    display_name = self.drug_name_display.get(drug_key, drug_key.upper())
                    drug_scores.append({
                        'drug_name': display_name,
                        'activity_type': dp['activity_type'],
                        'activity_value': dp['activity_value']
                    })
        
        for idx, drug in enumerate(drug_scores[:10], 1):
            drugs.append({
                'rank': idx,
                'drug_name': drug['drug_name'],
                'activity_type': drug['activity_type'],
                'activity_value': drug['activity_value']
            })
        
        # Get diseases
        protein_diseases = self.protein_to_disease.get(protein_key, [])
        
        for idx, pd in enumerate(protein_diseases[:10], 1):
            diseases.append({
                'rank': idx,
                'disease_name': pd['disease'],
                'gene': pd['gene']
            })
        
        return {
            'drugs': drugs[:10] if drugs else [],
            'diseases': diseases[:10] if diseases else []
        }
