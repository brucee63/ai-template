import pandas as pd
from ngram import NGram
import jellyfish
from rapidfuzz import fuzz

def ngram_match(user_input, customer_df, column_to_check, acronym_dict=None):
    """
    Perform n-gram matching between user input and DataFrame values, handling acronyms in values.
    
    Returns:
    - pd.DataFrame: DataFrame with n-gram scores and matched forms.
    """
    if column_to_check not in customer_df.columns:
        raise ValueError(f"Column '{column_to_check}' not found in DataFrame.")

    temp_df = customer_df.copy()
    if acronym_dict is None:
        acronym_dict = {}

    def ngram_similarity(name1, name2, n=3):
        return NGram.compare(name1, name2, N=n)

    def expand_acronyms(text, acronym_dict):
        variations = [text]
        words = text.split()
        for i, word in enumerate(words):
            if word in acronym_dict:
                expanded = acronym_dict[word]
                new_variation = " ".join(words[:i] + [expanded] + words[i+1:])
                variations.append(new_variation)
        return variations

    temp_df['ngram_score'] = 0.0
    temp_df['best_ngram_form'] = ""

    for index, row in temp_df.iterrows():
        original_value = row[column_to_check]
        value_variations = expand_acronyms(original_value, acronym_dict)
        
        best_ngram_score = 0.0
        best_form = original_value
        
        for variation in value_variations:
            score = ngram_similarity(user_input, variation, n=3)
            if score > best_ngram_score:
                best_ngram_score = score
                best_form = variation
        
        temp_df.at[index, 'ngram_score'] = best_ngram_score
        temp_df.at[index, 'best_ngram_form'] = best_form

    return temp_df[[column_to_check, 'ngram_score', 'best_ngram_form']]

def phonetic_match(user_input, customer_df, column_to_check, acronym_dict=None):
    """
    Perform phonetic matching between user input and DataFrame values, handling acronyms in values.
    
    Returns:
    - pd.DataFrame: DataFrame with phonetic match flags and matched forms.
    """
    if column_to_check not in customer_df.columns:
        raise ValueError(f"Column '{column_to_check}' not found in DataFrame.")

    temp_df = customer_df.copy()
    if acronym_dict is None:
        acronym_dict = {}

    def phonetic_similarity(name1, name2):
        soundex1 = jellyfish.soundex(name1)
        soundex2 = jellyfish.soundex(name2)
        return 1 if soundex1 == soundex2 else 0

    def expand_acronyms(text, acronym_dict):
        variations = [text]
        words = text.split()
        for i, word in enumerate(words):
            if word in acronym_dict:
                expanded = acronym_dict[word]
                new_variation = " ".join(words[:i] + [expanded] + words[i+1:])
                variations.append(new_variation)
        return variations

    temp_df['phonetic_match'] = 0
    temp_df['best_phonetic_form'] = ""

    for index, row in temp_df.iterrows():
        original_value = row[column_to_check]
        value_variations = expand_acronyms(original_value, acronym_dict)
        
        best_phonetic_score = 0
        best_form = original_value
        
        for variation in value_variations:
            score = phonetic_similarity(user_input, variation)
            if score > best_phonetic_score:
                best_phonetic_score = score
                best_form = variation
        
        temp_df.at[index, 'phonetic_match'] = best_phonetic_score
        temp_df.at[index, 'best_phonetic_form'] = best_form

    return temp_df[[column_to_check, 'phonetic_match', 'best_phonetic_form']]

def levenshtein_match(user_input, customer_df, column_to_check, acronym_dict=None):
    """
    Perform Levenshtein distance matching between user input and DataFrame values, handling acronyms.
    
    Returns:
    - pd.DataFrame: DataFrame with Levenshtein scores (0-1) and matched forms.
    """
    if column_to_check not in customer_df.columns:
        raise ValueError(f"Column '{column_to_check}' not found in DataFrame.")

    temp_df = customer_df.copy()
    if acronym_dict is None:
        acronym_dict = {}

    def levenshtein_similarity(name1, name2):
        return fuzz.ratio(name1, name2) / 100  # Normalize to 0-1

    def expand_acronyms(text, acronym_dict):
        variations = [text]
        words = text.split()
        for i, word in enumerate(words):
            if word in acronym_dict:
                expanded = acronym_dict[word]
                new_variation = " ".join(words[:i] + [expanded] + words[i+1:])
                variations.append(new_variation)
        return variations

    temp_df['levenshtein_score'] = 0.0
    temp_df['best_levenshtein_form'] = ""

    for index, row in temp_df.iterrows():
        original_value = row[column_to_check]
        value_variations = expand_acronyms(original_value, acronym_dict)
        
        best_levenshtein_score = 0.0
        best_form = original_value
        
        for variation in value_variations:
            score = levenshtein_similarity(user_input, variation)
            if score > best_levenshtein_score:
                best_levenshtein_score = score
                best_form = variation
        
        temp_df.at[index, 'levenshtein_score'] = best_levenshtein_score
        temp_df.at[index, 'best_levenshtein_form'] = best_form

    return temp_df[[column_to_check, 'levenshtein_score', 'best_levenshtein_form']]

def find_top_matches(user_input, customer_df, column_to_check, acronym_dict=None, top_n=5, method='hybrid'):
    """
    Find top matches using n-gram, phonetic, Levenshtein, or hybrid approaches.
    
    Parameters:
    - user_input (str): The input string to match.
    - customer_df (pd.DataFrame): DataFrame containing the data.
    - column_to_check (str): The column name to match against.
    - acronym_dict (dict, optional): Dictionary of acronyms to expanded forms.
    - top_n (int): Number of top results to return (default is 5).
    - method (str): 'hybrid' (default), 'ngram', 'phonetic', or 'levenshtein'.
    
    Returns:
    - pd.DataFrame: Top N matches with scores and match flags.
    """
    if method not in ['hybrid', 'ngram', 'phonetic', 'levenshtein']:
        raise ValueError("Method must be 'hybrid', 'ngram', 'phonetic', or 'levenshtein'.")

    if method == 'ngram':
        result_df = ngram_match(user_input, customer_df, column_to_check, acronym_dict)
        return result_df[[column_to_check, 'ngram_score']].sort_values(by='ngram_score', ascending=False).head(top_n)
    
    elif method == 'phonetic':
        result_df = phonetic_match(user_input, customer_df, column_to_check, acronym_dict)
        return result_df[[column_to_check, 'phonetic_match']].sort_values(by='phonetic_match', ascending=False).head(top_n)
    
    elif method == 'levenshtein':
        result_df = levenshtein_match(user_input, customer_df, column_to_check, acronym_dict)
        return result_df[[column_to_check, 'levenshtein_score']].sort_values(by='levenshtein_score', ascending=False).head(top_n)
    
    else:  # hybrid (default)
        ngram_df = ngram_match(user_input, customer_df, column_to_check, acronym_dict)
        phonetic_df = phonetic_match(user_input, customer_df, column_to_check, acronym_dict)
        
        result_df = ngram_df[[column_to_check, 'ngram_score']].merge(
            phonetic_df[[column_to_check, 'phonetic_match']],
            on=column_to_check,
            how='inner'
        )
        
        top_matches = (result_df[result_df['phonetic_match'] == 1]
                    .sort_values(by='ngram_score', ascending=False)
                    .head(top_n))
        return top_matches

# Example usage
if __name__ == "__main__":
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

    # Test with Levenshtein method
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='levenshtein')
    print(f"\nTop {len(top_matches)} Levenshtein matches for '{user_input}':")
    print(top_matches)