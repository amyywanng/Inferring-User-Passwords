############################ IMPORTS ############################ 
import os, re, glob, sys, nltk, spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Load SpaCy model for NER (open-source Python library for NLP)
try:
    nlp = spacy.load('en_core_web_sm')  # en - English, core - core spaCy pipeline, web - web-based training data, sm - small size
except:
    print("Installing required spaCy model...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


############################ CLASS DEFINITION (PasswordSeedGenerator) ############################ 
class PasswordSeedGenerator:
    def __init__(self, rcmp_data=None):
        """
        Initialize the PasswordSeedGenerator with RCMP data if available.
       
        Args:
            rcmp_data (dict): Dictionary containing personal information like:
                - name
                - birthdate
                - address
                - birthplace
                - family_members
                - height
                - weight
                - age
        """
        self.rcmp_data = rcmp_data or {}
        self.text_files = []
        self.raw_text = []  # file names
        self.token_counts = Counter()
        self.seeds = defaultdict(list)
        self.ranked_seeds = []
        self.known_passwords = []
        self.password_patterns = defaultdict(int)
       
        # Define priority categories for seeds
        self.category_priority = {
            'KNOWN_PASSWORD': 20,      # Highest priority for known passwords
            'RCMP': 15,                # RCMP data priority
            'WORDNET_GENERALIZED': 12, # Semantically related words
            'PERSON': 10,
            'DATE': 8,
            'GPE': 7,  # countries, cities, states
            'LOC': 7,   # non-GPE locations
            'ORG': 6,
            'PRODUCT': 5,
            'EVENT': 5,
            'WORK_OF_ART': 4,
            'FAC': 4,   # buildings, airports, highways, etc.
            'NORP': 3,  # nationalities, religious or political groups
            'LANGUAGE': 3,
            'OTHER': 1
        }

        # Initialize seeds with KNOWN_PASSWORD category
        self.seeds['KNOWN_PASSWORD'] = Counter()

        # Automatically analyze known passwords if file exists
        self.analyze_known_passwords("../known_passwords.txt")

    ############################ FUNCTION DEFINITIONS ############################ 

    ### NER (Named Entity Recognition) ###
    def identify_entities(self):
        """Use NER to identify entities in the text."""
        print("\nIdentifying named entities...")
       
        # Exclude weekday names and relative dates from DATE category
        excluded_dates = [
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'yesterday', 'today', 'tomorrow'
        ]
       
        for text in self.raw_text:
            # Tokenize and count for TF-IDF later
            tokens = word_tokenize(text.lower())
            self.token_counts.update(tokens)
           
            # Extract named entities
            doc = nlp(text[:1000000])  # Limit text size to avoid memory issues
            for ent in doc.ents:
                clean_ent = ent.text.replace(' ', '')
                if clean_ent:  # Only add non-empty entities
                    # Skip weekday names and relative dates in DATE category
                    if ent.label_ == 'DATE' and clean_ent.lower() in excluded_dates:
                        continue
                    
                    # Add the entity as a seed
                    self.seeds[ent.label_].append(clean_ent)
       
        # Convert lists to Counters (e.g. PERSON: ['John Smith', 'Mary Smith'] -> PERSON: Counter({'John Smith': 1, 'Mary Smith': 1}))
        for category, items in self.seeds.items():
            if isinstance(items, list):
                self.seeds[category] = Counter(items)
       
        print(f"Identified entities across {len(self.seeds)} categories")
        return self.seeds
   
    ### TF-IDF (Term Frequency-Inverse Document Frequency) ###
    def identify_important_terms(self, max_terms=100):
        """Use TF-IDF to identify important terms not caught by NER."""
        print("Identifying important terms using TF-IDF...")
       
        if not self.raw_text:
            print("No text available for TF-IDF analysis")
            return []
       
        # Vectorize the text
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_terms)  # creates a TF-IDF vectorizer that removes common English stopwords
        tfidf_matrix = vectorizer.fit_transform(self.raw_text)  # applies the vectorizer to the raw text to compute the TF-IDF matrix
       
        # Get feature names and their scores
        feature_names = vectorizer.get_feature_names_out()  # gets the feature names (words) from the vectorizer
        dense = tfidf_matrix.todense()  # converts the sparse TF-IDF matrix to a dense numpy array
        denselist = dense.tolist()  # converts the dense numpy array to a list of lists
       
        # Calculate average TF-IDF score for each term and stores them in a dictionary
        scores = {}
        for doc in denselist: # doc is a list of TF-IDF scores for a document
            for j, score in enumerate(doc): # score is the TF-IDF score for a term in the document
                term = feature_names[j] # uses j to look up the actual term (word) from the list of TF-IDF features
                if term in scores:
                    scores[term] = max(scores[term], score)  # updates the score for the term if it already exists, otherwise adds the term with its score
                else:
                    scores[term] = score
       
        # Store important terms
        important_terms = {term: score for term, score in scores.items()
                          if len(term) > 2 and score > 0}   # filters out very short or zero-scored terms to keep only meaningful ones
        sorted_terms = sorted(important_terms.items(), key=lambda x: x[1], reverse=True)  # sorts the terms by their TF-IDF scores in descending order
       
        # Add to seeds
        self.seeds['TFIDF'] = {term: score for term, score in sorted_terms}
       
        print(f"Identified {len(sorted_terms)} important terms")
        return sorted_terms
   
    ### WordNet (Semantic relationships between words) ###
    def generalize_with_wordnet(self):
        """Use WordNet to find semantically related words that aren't in the original documents."""
        print("\nGeneralizing words using WordNet semantic relationships...")
        
        try:
            from nltk.corpus import wordnet
        except ImportError:
            print("WordNet not available. Skipping semantic generalization.")
            return Counter()
        
        # Get all words from different extraction methods (NER and TF-IDF)
        existing_words = self._get_all_extracted_words()
        generalized_words = Counter()
        
        print(f"Analyzing {len(existing_words)} extracted words for semantic relationships...")
        
        for word in existing_words[:50]:  # Limit to avoid too much computation
            try:
                # Get WordNet synsets for the word
                synsets = wordnet.synsets(word.lower())
                
                if not synsets:
                    continue
                
                #print(f"  Analyzing '{word}' - found {len(synsets)} synsets")
                
                for synset in synsets[:3]:  # Limit to top 3 synsets per word
                    # Get hypernyms (more general terms)
                    for hypernym in synset.hypernyms():
                        for lemma in hypernym.lemmas():
                            generalized_word = lemma.name().replace('_', '')
                            if (len(generalized_word) > 3 and 
                                generalized_word not in existing_words and
                                generalized_word.isalpha()):
                                generalized_words[generalized_word] += 5
                                #print(f"    Hypernym: {word} -> {generalized_word}")
                    
                    # Get hyponyms (more specific terms)
                    for hyponym in synset.hyponyms():
                        for lemma in hyponym.lemmas():
                            generalized_word = lemma.name().replace('_', '')
                            if (len(generalized_word) > 3 and 
                                generalized_word not in existing_words and
                                generalized_word.isalpha()):
                                generalized_words[generalized_word] += 6
                                #print(f"    Hyponym: {word} -> {generalized_word}")
                    
                    # Get synonyms
                    for lemma in synset.lemmas():
                        generalized_word = lemma.name().replace('_', '')
                        if (generalized_word != word and 
                            len(generalized_word) > 3 and 
                            generalized_word not in existing_words and
                            generalized_word.isalpha()):
                            generalized_words[generalized_word] += 7
                            #print(f"    Synonym: {word} -> {generalized_word}")
                    
                    # Get related terms (meronyms, holonyms, etc.)
                    for related in synset.part_meronyms() + synset.substance_meronyms() + synset.member_meronyms():
                        for lemma in related.lemmas():
                            generalized_word = lemma.name().replace('_', '')
                            if (len(generalized_word) > 3 and 
                                generalized_word not in existing_words and
                                generalized_word.isalpha()):
                                generalized_words[generalized_word] += 4
                                #print(f"    Related: {word} -> {generalized_word}")
                                
            except Exception as e:
                print(f"    Error processing '{word}': {e}")
                continue
        
        # Add generalized words to seeds
        if generalized_words:
            self.seeds['WORDNET_GENERALIZED'] = generalized_words
            print(f"WordNet generalization complete: {len(generalized_words)} semantically related words found")
        else:
            print("No semantically related words found via WordNet")
        
        return generalized_words
    
    ### Get all words from different extraction methods (helper function for WordNet) ###
    def _get_all_extracted_words(self):
        """Get all words from different extraction methods."""
        all_words = set()  # set to avoid duplicates
        
        # Add words from NER entities
        for category in ['PERSON', 'ORG', 'PRODUCT', 'GPE', 'DATE']: # can add more categories here
            if category in self.seeds:
                all_words.update(self.seeds[category].keys())
        
        # Add words from TF-IDF
        if 'TFIDF' in self.seeds:
            all_words.update(self.seeds['TFIDF'].keys())
        
        # Add high-frequency words
        all_words.update([word for word, freq in self.token_counts.most_common(50)])
        
        # Add filenames
        if 'FILENAME' in self.seeds:
            all_words.update(self.seeds['FILENAME'])
        
        return list(all_words)
   
    ### Analyze known passwords to identify patterns and categories ###
    def analyze_known_passwords(self, password_file):
        """Analyze known passwords to identify patterns and categories."""
        print(f"Analyzing known passwords from {password_file}...")
        
        if not os.path.exists(password_file):
            print(f"Warning: {password_file} not found. Continuing without password analysis.")
            return
        
        try:
            with open(password_file, 'r', errors='ignore') as f:
                self.known_passwords = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading password file: {str(e)}")
            return
        
        # Define pattern categories
        patterns = {
            'dates': r'\d{4}|\d{2}',  # Years or dates
            'special_chars': r'[!@#$%^&*()_+\-=\[\]{};\'\\:"|,.<>/?]',
            'numbers': r'\d+',
            'uppercase': r'[A-Z]',
            'lowercase': r'[a-z]',
            'leetspeak': r'[4@3!1|0$5]',  # Common leetspeak substitutions
        }
        
        # Initialize or convert KNOWN_PASSWORD to Counter if needed
        if not isinstance(self.seeds['KNOWN_PASSWORD'], Counter):
            self.seeds['KNOWN_PASSWORD'] = Counter()
        
        # Analyze each password
        for password in self.known_passwords:
            # Store the full password with high weight
            self.seeds['KNOWN_PASSWORD'][password] = 80
            
            # Split password into components
            # Split by common separators and patterns
            components = []
            
            # Split by numbers
            number_parts = re.split(r'(\d+)', password)
            components.extend([p for p in number_parts if p])
            
            # Split by special characters
            special_parts = []
            for part in components:
                special_parts.extend(re.split(r'([!@#$%^&*()_+\-=\[\]{};\'\\:"|,.<>/?])', part))
            components = [p for p in special_parts if p]
            
            # Split by case changes (e.g., "JohnDoe" -> ["John", "Doe"])
            case_parts = []
            for part in components:
                if re.match(r'^[a-zA-Z]+$', part):  # Only split alphabetic parts
                    case_parts.extend(re.findall(r'[A-Z][a-z]*', part))
                else:
                    case_parts.append(part)
            components = [p for p in case_parts if p]
            
            # Add each component with high weight
            for component in components:
                if len(component) >= 2:  # Only store components of length 2 or more
                    self.seeds['KNOWN_PASSWORD'][component] = 60
            
            # Check for pattern matches
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, password):
                    self.password_patterns[pattern_name] += 1
            
            # Check for common password structures
            if re.match(r'^[A-Z][a-z]+\d+$', password):  # Capitalized word + numbers
                self.password_patterns['cap_word_num'] += 1
            elif re.match(r'^\d+[A-Z][a-z]+$', password):  # Numbers + capitalized word
                self.password_patterns['num_cap_word'] += 1
        
        print(f"Analyzed {len(self.known_passwords)} passwords")
        print(f"Extracted {len(self.seeds['KNOWN_PASSWORD'])} password components")
        print("Password patterns found:", dict(self.password_patterns))
        return self.password_patterns
   
    ### Compare with RCMP data ###
    def compare_with_rcmp_data(self):
        """Process RCMP data through NER and add to appropriate categories with higher priority."""
        if not self.rcmp_data:
            print("No RCMP data available for comparison")
            return
           
        print("\nProcessing RCMP data through NER...")
       
        # Process RCMP data through NER first
        rcmp_entities = defaultdict(list)
        
        # Process all RCMP data
        for category, value in self.rcmp_data.items():
            if isinstance(value, list): # if value is a list, handle each item in the list
                for item in value:
                    # Add raw RCMP data with RCMP label
                    clean_item = str(item).replace(' ', '')
                    if clean_item:
                        rcmp_entities['RCMP'].append(clean_item)
                    
                    # Process through NER
                    doc = nlp(str(item))
                    for ent in doc.ents:
                        clean_ent = ent.text.replace(' ', '')
                        if clean_ent:
                            # Only add to NER category, not to RCMP category
                            rcmp_entities[ent.label_].append(clean_ent)
            else:
                # Add raw RCMP data with RCMP label
                clean_value = str(value).replace(' ', '')
                if clean_value:
                    rcmp_entities['RCMP'].append(clean_value)
                
                # Process through NER
                doc = nlp(str(value))
                for ent in doc.ents:
                    clean_ent = ent.text.replace(' ', '')
                    if clean_ent:
                        # Only add to NER category, not to RCMP category
                        rcmp_entities[ent.label_].append(clean_ent)
        
        # Calculate dynamic weights based on document count
        doc_count = len(self.raw_text)
        if doc_count == 0:
            doc_count = 1  # Prevent division by zero
        
        # Calculate base weights dynamically
        # RCMP weight scales with document count but has a minimum value
        # Cap RCMP weight to be lower than KNOWN_PASSWORD weight (100)
        min_rcmp_weight = 50  # Minimum weight for RCMP data
        max_rcmp_weight = 90  # Maximum weight for RCMP data (below KNOWN_PASSWORD's 100)
        rcmp_base_weight = min(max_rcmp_weight, max(min_rcmp_weight, int(doc_count * 10)))  # Scale with doc count but cap at 90
        other_base_weight = max(25, int(rcmp_base_weight * 0.5))
        
        print(f"Dynamic weights - RCMP: {rcmp_base_weight}, Other: {other_base_weight} (based on {doc_count} documents)")
        
        # Add NER-processed RCMP entities to their respective categories
        for category, entities in rcmp_entities.items():
            # Initialize or convert to Counter if needed
            if category not in self.seeds:
                self.seeds[category] = Counter()
            elif not isinstance(self.seeds[category], Counter):
                self.seeds[category] = Counter(self.seeds[category])
                
            # Add entities to the Counter with higher priority for RCMP data
            for entity in entities:
                if category == 'RCMP':
                    # Give RCMP data a significant base weight
                    self.seeds[category][entity] = rcmp_base_weight
                else:
                    # Give other categories a moderate base weight
                    self.seeds[category][entity] = other_base_weight
        
        # Check for matches between RCMP entities and document entities
        matches = {}
        for category, items in self.seeds.items():
            if isinstance(items, Counter):
                for item, count in items.items():
                    if category in rcmp_entities:
                        if str(item).lower() in [e.lower() for e in rcmp_entities[category]]:
                            # Boost matches between RCMP and document entities
                            # Use a multiplier that scales with the base weight and document count
                            if category == 'RCMP':
                                # For RCMP matches, use a higher multiplier that scales with doc count
                                # Cap the final weight to be below KNOWN_PASSWORD weight
                                multiplier = min(3.0, 1.0 + (doc_count / 100))  # Cap at 3x
                                boosted_weight = max(count * multiplier, rcmp_base_weight * multiplier)
                                matches[item] = min(boosted_weight, max_rcmp_weight)  # Ensure we don't exceed max_rcmp_weight
                            else:
                                # For other matches, use a more moderate multiplier
                                multiplier = min(2.0, 1.0 + (doc_count / 200))  # Cap at 2x
                                matches[item] = max(count * multiplier, other_base_weight * multiplier)
            elif isinstance(items, dict):
                for item, score in items.items():
                    if category in rcmp_entities:
                        if str(item).lower() in [e.lower() for e in rcmp_entities[category]]:
                            # Apply same boosting logic for dict items
                            if category == 'RCMP':
                                multiplier = min(3.0, 1.0 + (doc_count / 100))
                                boosted_weight = max(score * multiplier, rcmp_base_weight * multiplier)
                                matches[item] = min(boosted_weight, max_rcmp_weight)  # Ensure we don't exceed max_rcmp_weight
                            else:
                                multiplier = min(2.0, 1.0 + (doc_count / 200))
                                matches[item] = max(score * multiplier, other_base_weight * multiplier)
        
        # Add matches back to their original categories with boosted scores
        for item, score in matches.items():
            for category, items in self.seeds.items():
                if isinstance(items, Counter) and item in items:
                    items[item] = score
                elif isinstance(items, dict) and item in items:
                    items[item] = score
        
        print(f"Found {len(matches)} matches between RCMP and document entities")
   
    ### Rank all identified seeds by priority and frequency ###
    def rank_seeds(self):
        """Rank all identified seeds by priority and frequency."""
        print("\nRanking password seeds...")
        ranked = []
       
        # Process each category
        for category, items in self.seeds.items():
            # Skip empty categories
            if not items:
                continue
               
            # Get priority for this category
            if category in self.category_priority:
                priority = self.category_priority[category]
            else:
                priority = self.category_priority.get('OTHER', 1)
           
            #print(f"Processing category: {category} (priority: {priority})")
           
            # Add all items with their scores
            if isinstance(items, Counter):
                for item, count in items.items():
                    score = priority * count
                    ranked.append((item, category, count, score))
            elif isinstance(items, dict):
                for item, value in items.items():
                    if category == 'TFIDF':
                        # Scale TF-IDF scores to be comparable
                        score = priority * value * 1000
                    else:
                        score = priority * value
                    ranked.append((item, category, value, score))
       
        # Sort by score (descending)
        self.ranked_seeds = sorted(ranked, key=lambda x: x[3], reverse=True)
       
        print(f"Ranked {len(self.ranked_seeds)} password seeds")
       
        return self.ranked_seeds
   
    def generate_passwords(self, num_passwords=100):
        """Generate password candidates from the ranked seeds."""
        print(f"Generating {num_passwords} password candidates...")
       
        if not self.ranked_seeds:
            print("No seeds available to generate passwords")
            return []
       
        passwords = set()
       
        # Helper function to generate variations
        def add_variations(seed, category):
            # Clean the seed string by removing commas, spaces, backslashes and forward slashes
            seed_str = str(seed).replace(',', '').replace(' ', '').replace('\\', '').replace('/', '')
            variations = [
                seed_str,                             # Original
                seed_str.lower(),                     # lowercase
                seed_str.upper(),                     # UPPERCASE
                seed_str.capitalize(),                # Capitalized
                re.sub(r'[aeiou]', '', seed_str),     # Remove vowels
                re.sub(r'[^\w]', '', seed_str)        # Remove non-alphanumeric
            ]
           
            # Add common number suffixes
            for year in range(1970, 2023):
                variations.append(f"{seed_str}{year}")  # End
                variations.append(f"{year}{seed_str}")  # Beginning
           
            for num in ['123', '1234', '12345', '123456', '1', '01', '1!', '!']:
                variations.append(f"{seed_str}{num}")  # End
                variations.append(f"{num}{seed_str}")  # Beginning
           
            # Add special character variations
            for char in ['!', '@', '#', '$', '%', '&', '*']:
                variations.append(f"{seed_str}{char}")  # End
                variations.append(f"{char}{seed_str}")  # Beginning
                variations.append(f"{seed_str}{char}{char}")  # End double
                variations.append(f"{char}{char}{seed_str}")  # Beginning double
               
            # Leet speak replacements
            leet_seed = seed_str
            leet_replacements = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7'}
            for char, replacement in leet_replacements.items():
                leet_seed = leet_seed.replace(char, replacement)
                leet_seed = leet_seed.replace(char.upper(), replacement)
            variations.append(leet_seed)
           
            # Add all variations to the set
            for var in variations:
                if var and len(var) >= 6:  # Only add reasonably long passwords
                    passwords.add(var)
       
        # Generate variations of top seeds
        top_seeds = self.ranked_seeds[:min(50, len(self.ranked_seeds))]
        
        for seed, category, count, score in top_seeds:
            add_variations(seed, category)
       
        # Generate combinations of top seeds
        for i, (seed1, _, _, _) in enumerate(top_seeds):
            for j, (seed2, _, _, _) in enumerate(top_seeds):
                if i != j:
                    # Clean both seeds before combining
                    clean_seed1 = str(seed1).replace(',', '').replace(' ', '').replace('\\', '').replace('/', '')
                    clean_seed2 = str(seed2).replace(',', '').replace(' ', '').replace('\\', '').replace('/', '')
                    
                    # Basic combination
                    combination = f"{clean_seed1}{clean_seed2}"
                    if len(combination) >= 6:
                        passwords.add(combination)
                        passwords.add(combination.lower())
                        passwords.add(combination.capitalize())
                    
                    # Add numbers between seeds
                    for year in range(1970, 2023):
                        passwords.add(f"{clean_seed1}{year}{clean_seed2}")
                    
                    for num in ['123', '1234', '12345', '123456', '1', '01', '1!', '!']:
                        passwords.add(f"{clean_seed1}{num}{clean_seed2}")
                    
                    # Add special characters between seeds
                    for char in ['!', '@', '#', '$', '%', '&', '*']:
                        passwords.add(f"{clean_seed1}{char}{clean_seed2}")
                        passwords.add(f"{clean_seed1}{char}{char}{clean_seed2}")
       
        # Convert set to sorted list (limit to requested number)
        password_list = list(passwords)[:num_passwords]
       
        print(f"Generated {len(password_list)} unique password candidates")
        return password_list
   
    ### Scan a directory for text-based files ###
    def scan_directory(self, directory):
        """Scan a directory for text-based files."""
        print(f"Scanning directory: {directory}")
        text_extensions = ['.txt', '.doc', '.docx', '.pdf', '.md', '.csv',
                          '.rtf', '.log', '.json', '.xml', '.html', '.py',
                          '.js', '.c', '.cpp', '.java', '.h', '.sql']
       
       # Using glob to search for all files in a directory (and its subdirectories) that match the file extensions
        for ext in text_extensions:
            pattern = os.path.join(directory, f"**/*{ext}")
            self.text_files.extend(glob.glob(pattern, recursive=True))
       
        print(f"Found {len(self.text_files)} text files")
        return self.text_files
   
    ### Extract text from found files ###
    def extract_text(self, max_files=None):
        """Extract text from found files."""
        if max_files:
            files_to_process = self.text_files[:max_files]
        else:
            files_to_process = self.text_files
           
        print(f"Extracting text from {len(files_to_process)} files")
       
        for file_path in files_to_process:
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    text = f.read()
                    # Clean the text by removing extra spaces
                    text = ' '.join(text.split())
                    self.raw_text.append(text)
                   
                    # Add filename as potential seed
                    filename = os.path.basename(file_path)
                    name, _ = os.path.splitext(filename)
                    if name.lower() != 'password':
                        self.seeds['FILENAME'].append(name.replace(' ', ''))
                       
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
       
        print(f"Extracted text from {len(self.raw_text)} files", end='\n')
        return self.raw_text

    ############################################################################################
    
    ### Run the complete password generation pipeline ###
    def run_pipeline(self, directory, max_files=None, num_passwords=100):
        """Run the complete password generation pipeline."""
        print("Starting password generation pipeline...")
       
        # Step 1: Scan for files
        self.scan_directory(directory)
       
        # Step 2: Extract text from files
        self.extract_text(max_files)
       
        # Step 3: Identify entities using NER
        self.identify_entities()
       
        # Step 4: Identify important terms using TF-IDF
        self.identify_important_terms()
       
        # Step 5: Generalize with WordNet
        self.generalize_with_wordnet()
       
        # Step 6: Compare with RCMP data
        self.compare_with_rcmp_data()
       
        # Step 7: Rank seeds
        self.rank_seeds()
       
        # Step 8: Generate passwords
        passwords = self.generate_passwords(num_passwords)
       
        print("Password generation pipeline completed")
        return {
            'seeds': self.ranked_seeds,
            'passwords': passwords,
            'wordnet_generalized': self.seeds.get('WORDNET_GENERALIZED', Counter())
        } 