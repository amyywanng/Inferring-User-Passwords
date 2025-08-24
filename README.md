# Password Seed Generator

This project implements a sophisticated password seed identification and generation system. It analyzes text files to identify potential password seeds based on named entities, important terms, WordNet semantic generalization, known password patterns, and RCMP data.

## Features

- Scans multiple file types (txt, doc, docx, pdf, md, csv, etc.)
- Uses Named Entity Recognition (NER) to identify potential password seeds
- Implements TF-IDF analysis to find important terms
- **WordNet semantic generalization** to discover semantically related words not present in documents
- **Known password pattern analysis** to identify proven password structures and components
- Integrates with RCMP data to boost relevant seeds
- Generates password variations using common patterns and transformations
- Supports combination of multiple seeds
- Ranks seeds by importance and frequency
- Filters out less meaningful date entities (weekdays, relative dates)
- Balances RCMP data influence for more diverse passwords
- **Dynamic weight scaling** based on document count to prevent over-prioritization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install the required spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. Enter the directory to scan when prompted (or press Enter to use the current directory)

3. The script will:
   - Scan the specified directory for text files
   - Extract and analyze text from these files
   - Identify potential password seeds using NER and TF-IDF
   - **Generate semantically related words using WordNet**
   - **Analyze known password patterns from reference data**
   - Generate password variations
   - Save results to:
     - `password_seeds.txt`: Contains ranked password seeds
     - `password_candidates.txt`: Contains generated password candidates
     - `wordnet_analysis.txt`: Contains WordNet semantic generalization results

## RCMP Data Integration

The system can integrate with RCMP data to boost the importance of relevant seeds. The RCMP data should be provided as a dictionary with the following structure:

```python
rcmp_data = {
    'name': 'John Smith',
    'birthdate': '1985-06-15',
    'address': '123 Main St, Ottawa',
    'birthplace': 'Toronto',
    'family_members': ['Mary Smith', 'Emma Smith', 'James Smith'],
    'height': '180cm',
    'weight': '75kg',
    'age': 38
}
```

### RCMP Data Processing

The system processes RCMP data through several steps:

1. **NER Processing**: RCMP data is processed through Named Entity Recognition to identify entities like PERSON, DATE, GPE, etc.

2. **Category Assignment**: Entities are added to their respective NER categories with a boost factor to indicate they're from RCMP data.

3. **Dynamic Weight Scaling**: RCMP weights scale with document count (50-90 range) to prevent over-prioritization in data-rich environments.

4. **Match Boosting**: When RCMP entities match with document entities, they receive a higher boost factor (up to 3x for RCMP, 2x for others).

5. **Duplicate Prevention**: The system avoids adding the same entity to multiple categories to prevent duplicates in the ranked seeds.

## WordNet Semantic Generalization

The system uses WordNet to discover semantically related words that aren't explicitly present in the original documents:

1. **Word Collection**: Gathers words from NER entities, TF-IDF terms, high-frequency words, and filenames
2. **Semantic Analysis**: Processes the top 50 extracted words through WordNet's semantic network
3. **Relationship Types**:
   - **Synonyms** (weight +7): Direct synonyms of extracted words
   - **Hyponyms** (weight +6): More specific terms
   - **Hypernyms** (weight +5): More general terms  
   - **Related Terms** (weight +4): Meronyms, holonyms, and other semantic relationships
4. **Filtering**: Only includes words >3 characters, not in original documents, and alphabetic only
5. **Output**: Results saved to `wordnet_analysis.txt` with detailed relationship breakdown

## Known Password Pattern Analysis

The system analyzes a reference file of known passwords to identify common patterns:

1. **Pattern Recognition**: Identifies structural patterns like capitalized words + numbers, special character usage, and leet speak
2. **Component Extraction**: Decomposes passwords into individual components (words, numbers, special characters)
3. **Weight Assignment**: 
   - Full passwords: weight 80
   - Password components: weight 60
4. **Pattern Categories**: Dates, special characters, numbers, case variations, leet speak substitutions
5. **Integration**: Known patterns inform password generation strategies and receive high priority in ranking

## Entity Handling

The system processes entities through several steps:

1. **Entity Identification**: Uses NER to identify entities in the document text.

2. **Entity Filtering**:
   - Excludes weekday names ('monday', 'tuesday', etc.) from the DATE category
   - Excludes relative dates ('yesterday', 'today', 'tomorrow') from the DATE category

3. **Entity Categorization**: Entities are categorized based on their NER label (PERSON, DATE, GPE, etc.).

4. **Entity Prioritization**: Each category has a priority value:
   - **KNOWN_PASSWORD**: 20 (highest)
   - **RCMP**: 15
   - **WORDNET_GENERALIZED**: 12
   - **PERSON**: 10
   - **DATE**: 8
   - **GPE/LOC**: 7
   - **ORG**: 6
   - **PRODUCT/EVENT**: 5
   - **WORK_OF_ART/FAC**: 4
   - **NORP/LANGUAGE**: 3
   - **OTHER**: 1 (lowest)

## Password Generation Features

The system generates passwords using various techniques:

1. **Basic Variations**:
   - Original form
   - Lowercase
   - Uppercase
   - Capitalized
   - Vowel removal
   - Non-alphanumeric removal

2. **Number Insertions**:
   - Beginning: `{num}{seed}`
   - End: `{seed}{num}`
   - Middle (for combinations): `{seed1}{num}{seed2}`
   - Years (1970-2023)
   - Common numbers (123, 1234, etc.)

3. **Special Character Insertions**:
   - Beginning: `{char}{seed}`
   - End: `{seed}{char}`
   - Middle (for combinations): `{seed1}{char}{seed2}`
   - Double special characters: `{seed1}{char}{char}{seed2}`
   - Common special characters (!, @, #, $, %, &, *)

4. **Leet Speak**:
   - Common character replacements (a→4, e→3, i→1, o→0, s→5, t→7)

5. **Combinations**:
   - Pairs of top-ranked seeds (increased from 20 to 50 seeds)
   - Various case combinations
   - Number and special character insertions

## Output Format

### Password Seeds File
```
Password Seeds:
1. JohnSmith (KNOWN_PASSWORD) - Score: 2000.00
2. developer (WORDNET_GENERALIZED) - Score: 84.00
3. 19850615 (DATE) - Score: 16.00
4. Toronto (GPE) - Score: 14.00
...
```

### Password Candidates File
```
Password Candidates:
JohnSmith
johnsmith
JOHNSMITH
JohnSmith123
JohnSmith1985
JohnSmith!
JohnSmith@
developer123
developer1985
...
```

### WordNet Analysis File
```
WordNet Semantic Generalization Analysis
==================================================

Semantically related words found (not in original documents):
  programmer: 7
  coder: 7
  creator: 7
  team: 4
  project: 4
...
```

## Recent Improvements

1. **WordNet Semantic Generalization**:
   - Added semantic relationship discovery beyond literal document content
   - Implemented weighted scoring for different relationship types
   - Limited processing to top 50 words for performance
   - Added detailed output to `wordnet_analysis.txt`

2. **Known Password Pattern Analysis**:
   - Added analysis of reference password file
   - Implemented pattern recognition for common password structures
   - Assigned high priority weights to proven password components
   - Enhanced password generation with pattern-based strategies

3. **Enhanced RCMP Data Processing**:
   - Implemented dynamic weight scaling based on document count
   - Adjusted boost factors to prevent over-prioritization
   - Improved match detection between RCMP and document entities

4. **Code Optimization**:
   - Increased top seeds used for password generation from 20 to 50
   - Adjusted TF-IDF scaling factor for better score comparability
   - Removed unused code and improved efficiency
   - Enhanced debugging output for WordNet analysis

5. **File Type Support**:
   - Expanded support to 15+ text-based file formats
   - Improved filename extraction and processing

## Customization

You can customize various aspects of the system by modifying:

1. `category_priority` in `PasswordSeedGenerator.__init__()` to adjust seed category weights
2. File type extensions in `scan_directory()`
3. Password generation patterns in `generate_passwords()`
4. TF-IDF parameters in `identify_important_terms()`
5. **WordNet processing limits** in `generalize_with_wordnet()`
6. **Known password file path** in `analyze_known_passwords()`

## Requirements

- Python 3.7+
- spaCy with en_core_web_sm model
- NLTK with WordNet corpus
- scikit-learn for TF-IDF analysis
- Standard Python libraries (os, re, glob, collections)

## File Structure

```
src/
├── main.py                           # Main execution script
├── password_seed_generator.py        # Core password generation logic
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── known_passwords.txt              # Reference password patterns (optional)

Output files:
├── password_seeds.txt               # Ranked password seeds
├── password_candidates.txt          # Generated password candidates
└── wordnet_analysis.txt            # WordNet semantic analysis results
``` 