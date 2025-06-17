# Adaptive Multilingual Cyberbullying Detection via Grammatical Evolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Advanced framework for detecting cyberbullying across languages using grammatical evolution and transformer models. Supports English and Arabic social media text with dialect normalization.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Contributing](#contributing)
- [Citation](#citation)

## Features <a name="features"></a>
- 🌐 **Multilingual Support**: Handles English and Arabic with dialect normalization
- 🧬 **Evolutionary Architecture Search**: Optimizes neural networks using DEAP framework
- ⚖️ **Fairness-Aware Validation**: Demographic parity constraints for equitable performance
- 📊 **Comprehensive Metrics**: Accuracy, F1, Precision, Recall + Confusion Matrices
- 🤖 **Multiple Architectures**: MLP, RNN, LSTM, BiLSTM with dynamic composition
- 🔄 **Real-Time Adaptation**: Lightweight retraining for emerging bullying patterns

## Installation <a name="installation"></a>

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/wasay530/Adaptive-Multilingual-Cyberbullying-Detection-via-Grammatical-Evolution.git
cd cyberbullying-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt

Requirements:
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- DEAP 1.3+
- Pandas 1.5+
- Scikit-learn 1.2+
```

## File Structure
```bash
data/
├── ArCyC A Fully Annotated Arabic Cyberbullying Corpus/
│   ├── training_data.csv
│   ├── valid_data.csv
│   └── testing_data.csv
└── ArCyC A Fully Annotated Arabic Cyberbullying Corpus/
    ├── training_data.csv
    ├── valid_data.csv
    └── testing_data.csv
```

## Dataset Setup <a name="dataset-setup"></a>

## CSV Format
| Column | Description | Values |
| ------------- | ------------- | ------------- |
| text | Social media text | Raw string |
| gender | User gender | male, female, other |
| age_group | User age | 13-17, 18-25, 26-35, 36+ |
| religion | User religion | christian, muslim, jewish, other |

## Sample Entry (Arabic):

``` "text":"أنت hater مثير للاشمئزاز","label":1,"gender":"male","age_group":"18-25","religion":"muslim" ```

## Usage <a name="usage"></a>
Basic Execution
```bash
# English dataset with default parameters
python main.py --dataset english

# Arabic dataset with custom evolution
python main.py --dataset arabic --pop_size 500 --generations 50
```

Command-line Options
| Parameter | Description | Default |
| ------------- | ------------- | ------------- |
| dataset | Dataset to use (english/arabic)	| english |
| runs	| Number of evolutionary runs	| 30 |
| pop_size	| Population size	| 1000 |
| generations	| Evolution generations	| 100 |
| device	| Compute device (cpu/cuda)	| auto-detect |

## Export Model
```bash
from models.factory import ModelFactory

best_config = {...}  # From evolution results
model = ModelFactory.create(best_config, input_size=768, output_size=2)
torch.onnx.export(model, torch.randn(1,768), "model.onnx")
```

## Project Structure <a name="project-structure"></a>
```bash
├── config/               
│   ├── settings.py      
│   └── grammar.py       
├── data/               
│   ├── data_loader.py    
│   ├── text_cleaner.py  
│   └── embeddings.py    
├── models/               
│   ├── mlp.py           
│   ├── lstm.py          
│   └── factory.py       
├── evolution/           
│   ├── individual.py   
│   └── optimizer.py     
├── evaluation/          
│   ├── metrics.py       
│   └── fairness.py      
└── main.py              
```

## Methodology <a name="methodology"></a>
1. Text Preprocessing
  * Language detection with langdetect
  * Dialect normalization (Gulf → MSA Arabic)
  * BERT/XLM-R embeddings + TF-IDF n-grams
2. Evolutionary Optimization
```bash
GRAMMAR = {
    'model_type': ['mlp', 'lstm', 'bilstm', 'rnn'],
    'layers': [2, 3, 4],
    'hidden_size': [128, 256, 512],
    'dropout': [0.2, 0.3, 0.4]
}
```
  * Population: 1000 individuals
  * Generations: 100
  * Selection: Tournament (size=3)
  * Crossover: 70% probability
  * Mutation: 15% probability

3. Fairness Constraints
  * Maximum performance disparity: 5%
  * Protected attributes: Gender, Age, Religion  
  * Penalty formula: fitness = accuracy - (max_disparity * 0.2)

## Contributing <a name="contributing"></a>
* Fork the repository
* Create feature branch:
```git checkout -b feature/new-feature```
* Commit changes:
```git commit -am 'Add new feature'```
* Push to branch:
```git push origin feature/new-feature```
* Open a Pull Request

## Citation <a name="citation"></a>
```bash
@software{cyberbullying_detection_2025,
  author = {Your Name},
  title = {Adaptive Multilingual Cyberbullying Detection via Grammatical Evolution},
  year = {2025},
  publisher = {PeerJ},
  journal = {PeerJ},
  url = {https://github.com/wasay530/Adaptive-Multilingual-Cyberbullying-Detection-via-Grammatical-Evolution}
}
```
