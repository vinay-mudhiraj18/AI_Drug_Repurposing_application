# AI-Powered Drug Repurposing Platform

A Django-based web application that leverages machine learning and knowledge graph analysis to identify potential new therapeutic uses for existing drugs. The platform is designed to support researchers and healthcare professionals in accelerating drug discovery and reducing development costs.

---

## Overview

Drug development is a time-consuming and expensive process, often taking over a decade and billions of dollars to bring a single drug to market. Drug repurposing offers a more efficient alternative by identifying new uses for approved or existing drugs.

This platform applies a knowledge graph-driven approach to model relationships between drugs, proteins, and diseases, enabling both validated and predictive insights.

---

## Key Features

* **Multi-Entity Search**

  * Drug-based queries
  * Disease-based queries
  * Protein-based queries

* **Two-Tier Prediction System**

  * **Validated Results**: Clinically proven drug-disease relationships
  * **Predicted Results**: Machine learning-based associations using protein networks

* **Interactive User Interface**

  * Structured and ranked outputs
  * Search history tracking
  * Responsive design

* **Administrative Dashboard**

  * User management
  * Query analytics
  * System statistics

---

## System Architecture

The platform is built on a knowledge graph that models relationships as:

```text
Drug → Protein → Disease
```

### Prediction Workflow

1. Input validation and normalization
2. Lookup in validated therapeutic dataset
3. Protein target identification
4. Graph traversal across protein-disease relationships
5. Similarity computation using vector embeddings
6. Ranking based on confidence scores

---

## Technology Stack

### Backend

* Django
* Python
* SQLite
* Pandas, NumPy
* Scikit-learn

### Frontend

* HTML5, CSS3
* JavaScript
* Chart.js

---

## Installation

### Prerequisites

* Python 3.8 or higher
* pip

### Setup

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Access

* Application: http://127.0.0.1:8000/
* Admin Dashboard: http://127.0.0.1:8000/admin-dashboard/

---

## Data Sources

The system integrates multiple biomedical datasets, including:

* Drug–protein interaction data
* Disease–gene association datasets
* Validated therapeutic relationships
* Precomputed embedding vectors for similarity analysis

---

## Performance

* Average query response time: < 0.5 seconds
* Optimized data structures for fast lookups
* Efficient preprocessing and deduplication

---

## Security

* CSRF protection
* Secure authentication using Django’s built-in system
* Input validation and sanitization
* Session management

---

## Use Cases

* Identification of alternative treatments for known diseases
* Exploration of drug-protein interactions
* Supporting research in computational drug discovery
* Academic and educational purposes

---

## Disclaimer

This platform is intended for research and educational use only.

Predicted results are generated using computational models and should not be considered medical advice. Always consult qualified healthcare professionals before making clinical decisions.

---

## Future Enhancements

* Integration with external biomedical APIs
* Real-time data updates
* Advanced deep learning models
* Visualization of knowledge graphs

---

## License

This project is for academic and research purposes. Licensing terms can be updated based on future use.

---

## Contact

For questions or collaboration opportunities, please reach out via GitHub.

---
