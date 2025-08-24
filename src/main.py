from password_seed_generator import PasswordSeedGenerator


def main():
    """Main function to demonstrate usage."""
    # Sample RCMP data (in a real scenario, this would be provided)
    
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
   
    # Initialize the generator (calling class from other file)
    generator = PasswordSeedGenerator(rcmp_data)
   
    # Run the pipeline on a specified directory
    directory = input("Enter the directory to scan (default: current directory): ") or '.'
    results = generator.run_pipeline(directory, max_files=100, num_passwords=200)
  
   # Display WordNet generalized words
    if results['wordnet_generalized']:
        print(f"\nWordNet Generalized Words ({len(results['wordnet_generalized'])} found):")
        for word, count in results['wordnet_generalized'].most_common(15):
            print(f"  {word}: {count}")
    
    # Display top seeds
    print("\nTop 20 Password Seeds:")
    for i, (seed, category, count, score) in enumerate(results['seeds'][:20], 1):
        print(f"{i}. {seed} ({category}) - Score: {score:.2f}")
    
    # Display top password candidates
    print("\nTop 20 Password Candidates:")
    for i, password in enumerate(results['passwords'][:20], 1):
        print(f"{i}. {password}")
    
    
    # Save results to file
    with open('password_seeds.txt', 'w') as f:
        f.write("Password Seeds:\n")
        for seed, category, count, score in results['seeds']:
            f.write(f"{seed} ({category}) - Score: {score:.2f}\n")
    
    with open('password_candidates.txt', 'w') as f:
        f.write("Password Candidates:\n")
        for password in results['passwords']:
            f.write(f"{password}\n")
    
    # Save WordNet analysis
    if results['wordnet_generalized']:
        with open('wordnet_analysis.txt', 'w') as f:
            f.write("WordNet Semantic Generalization Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write("Semantically related words found (not in original documents):\n")
            for word, count in results['wordnet_generalized'].most_common():
                f.write(f"  {word}: {count}\n")
    
    print("\nResults saved to password_seeds.txt, password_candidates.txt, and wordnet_analysis.txt")


if __name__ == "__main__":
    main() 