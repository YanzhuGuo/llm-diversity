# LLM Diversity Analysis

This repository provides code for the paper **Benchmarking Linguistic Diversity of Large Language Models**

## Repository Structure

### Modules
- **MT (Machine Translation):**
  - `code/translate.py` : Generates translations
- **LM (Language Modeling):**
  - `code/wiki.py` : Generates blocks of text continuation
- **Summ (Summarization):**
  - `code/summary.py` : Generates summaries
- **ASG (Automatic Story Generation):**
  - `code/story.py` : Generates stories
- **NUG (Dialogue Generation):**
  - `code/dialogue.py` : Generates next utterances

### Diversity Metrics
Located in `diversity_metrics/`:
- `lexical_diversity.py`: Evaluates lexical diversity
- `syntactic_diversity.py`: Evaluates syntactic diversity
- `semantic_diversity.py`: Evaluates semantic diversity
