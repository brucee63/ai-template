import pandas as pd
from utils.matching import find_top_matches

def test_matching():
    # Sample data with acronyms in the values
    data = {
        'full_name': [
            'JS Plumbing',          # "JS" = "John Smith"
            'Jon Smyth Plumbing',
            'JB Electrical',        # "JB" = "James Brown"
            'Jim Browne Electrical',
            'CJ Bakery',            # "CJ" = "Catherine Jones"
            'Kathryn Jons Bakery',
            'Jonah Smithers Plumbing'
        ]
    }
    df = pd.DataFrame(data)

    # Acronym dictionary
    acronym_dict = {
        'JS': 'John Smith',
        'JB': 'James Brown',
        'CJ': 'Catherine Jones'
    }

    # Test with hybrid method
    user_input = "John Smith Plumbing"
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='hybrid')
    print(f"Top {len(top_matches)} hybrid matches for '{user_input}':")
    print(top_matches)

    assert len(top_matches) > 0
    assert top_matches.iloc[0]['full_name'] == 'JS Plumbing'  

    # Test with Levenshtein method
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='levenshtein')
    print(f"\nTop {len(top_matches)} Levenshtein matches for '{user_input}':")
    print(top_matches)

    assert len(top_matches) > 0
    assert top_matches.iloc[0]['full_name'] == 'JS Plumbing'  

    # Test with jarowinkler method
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='jarowinkler')
    print(f"\nTop {len(top_matches)} Jaro-Winkler matches for '{user_input}':")
    print(top_matches)
    
    assert len(top_matches) > 0
    assert top_matches.iloc[0]['full_name'] == 'JS Plumbing'
    
    # Test with n-gram method
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='ngram')
    print(f"\nTop {len(top_matches)} n-gram matches for '{user_input}':")
    print(top_matches)
    
    assert len(top_matches) > 0
    assert top_matches.iloc[0]['full_name'] == 'JS Plumbing'

    # Test with jaccard method
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='jaccard')
    print(f"\nTop {len(top_matches)} Jaccard matches for '{user_input}':")
    print(top_matches)