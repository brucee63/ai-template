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

def jaro_winkler_match(user_input, customer_df, column_to_check, acronym_dict=None):
    """
    Perform Jaro-Winkler similarity matching between user input and DataFrame values, handling acronyms.
    
    Args:
        user_input (str): The input string to match against.
        customer_df (pd.DataFrame): DataFrame containing the data to match against.
        column_to_check (str): Column in the DataFrame to perform matching on.
        acronym_dict (dict, optional): Dictionary mapping acronyms to their expanded forms.
    
    Returns:
        pd.DataFrame: DataFrame with Jaro-Winkler scores (0-1) and matched forms.
    """
    if column_to_check not in customer_df.columns:
        raise ValueError(f"Column '{column_to_check}' not found in DataFrame.")

    temp_df = customer_df.copy()
    if acronym_dict is None:
        acronym_dict = {}

    def jaro_winkler_similarity(s1, s2):
        # Handle empty strings
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        # Find matching characters and their pairs
        max_distance = (max(len(s1), len(s2)) // 2) - 1
        possible_j_for_i = [[] for _ in range(len(s1))]
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j] and abs(i - j) <= max_distance:
                    possible_j_for_i[i].append(j)

        # Find matching pairs (greedy approach)
        used_j = set()
        matching_pairs = []
        for i in range(len(s1)):
            for j in possible_j_for_i[i]:
                if j not in used_j:
                    used_j.add(j)
                    matching_pairs.append((i, j))
                    break

        m = len(matching_pairs)
        if m == 0:
            return 0.0

        # Calculate transpositions (t)
        t = sum(1 for i, j in matching_pairs if i != j) / 2.0

        # Calculate Jaro similarity
        jaro = (m / len(s1) + m / len(s2) + (m - t) / m) / 3.0

        # Calculate common prefix length (up to 4)
        l = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                l += 1
            else:
                break

        # Calculate Jaro-Winkler similarity with p=0.1
        p = 0.1
        jaro_winkler = jaro + l * p * (1 - jaro)

        return jaro_winkler

    def expand_acronyms(text, acronym_dict):
        variations = [text]
        words = text.split()
        for i, word in enumerate(words):
            if word in acronym_dict:
                expanded = acronym_dict[word]
                new_variation = " ".join(words[:i] + [expanded] + words[i+1:])
                variations.append(new_variation)
        return variations

    temp_df['jaro_winkler_score'] = 0.0
    temp_df['best_jaro_winkler_form'] = ""

    for index, row in temp_df.iterrows():
        original_value = str(row[column_to_check])  # Ensure string type
        value_variations = expand_acronyms(original_value, acronym_dict)
        
        best_jaro_winkler_score = 0.0
        best_form = original_value
        
        for variation in value_variations:
            score = jaro_winkler_similarity(user_input, variation)
            if score > best_jaro_winkler_score:
                best_jaro_winkler_score = score
                best_form = variation
        
        temp_df.at[index, 'jaro_winkler_score'] = best_jaro_winkler_score
        temp_df.at[index, 'best_jaro_winkler_form'] = best_form

    return temp_df[[column_to_check, 'jaro_winkler_score', 'best_jaro_winkler_form']]

def jaccard_match(user_input, customer_df, column_to_check, acronym_dict=None):
    """
    Perform Jaccard similarity matching between user input and DataFrame values, handling acronyms.
    
    Args:
        user_input (str): The input string to match against.
        customer_df (pd.DataFrame): DataFrame containing the data to match against.
        column_to_check (str): Column in the DataFrame to perform matching on.
        acronym_dict (dict, optional): Dictionary mapping acronyms to their expanded forms.
    
    Returns:
        pd.DataFrame: DataFrame with Jaccard scores (0-1) and matched forms.
    """
    if column_to_check not in customer_df.columns:
        raise ValueError(f"Column '{column_to_check}' not found in DataFrame.")

    temp_df = customer_df.copy()
    if acronym_dict is None:
        acronym_dict = {}

    def jaccard_similarity(s1, s2):
        # Handle empty strings
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        # Convert strings to sets of words
        set1 = set(s1.split())
        set2 = set(s2.split())
        
        # Calculate intersection and union
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        # Jaccard similarity is intersection over union
        return intersection / union if union > 0 else 0.0

    def expand_acronyms(text, acronym_dict):
        variations = [text]
        words = text.split()
        for i, word in enumerate(words):
            if word in acronym_dict:
                expanded = acronym_dict[word]
                new_variation = " ".join(words[:i] + [expanded] + words[i+1:])
                variations.append(new_variation)
        return variations

    temp_df['jaccard_score'] = 0.0
    temp_df['best_jaccard_form'] = ""

    for index, row in temp_df.iterrows():
        original_value = str(row[column_to_check])  # Ensure string type
        value_variations = expand_acronyms(original_value, acronym_dict)
        
        best_jaccard_score = 0.0
        best_form = original_value
        
        for variation in value_variations:
            score = jaccard_similarity(user_input, variation)
            if score > best_jaccard_score:
                best_jaccard_score = score
                best_form = variation
        
        temp_df.at[index, 'jaccard_score'] = best_jaccard_score
        temp_df.at[index, 'best_jaccard_form'] = best_form

    return temp_df[[column_to_check, 'jaccard_score', 'best_jaccard_form']]

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
    if method not in ['hybrid', 'ngram', 'phonetic', 'levenshtein', 'jarowinkler', 'jaccard']:
        raise ValueError("Method must be 'hybrid', 'ngram', 'phonetic', 'levenshtein', 'jarowinkler', or 'jaccard'.")

    if method == 'ngram':
        result_df = ngram_match(user_input, customer_df, column_to_check, acronym_dict)
        return result_df[[column_to_check, 'ngram_score']].sort_values(by='ngram_score', ascending=False).head(top_n)
    
    elif method == 'phonetic':
        result_df = phonetic_match(user_input, customer_df, column_to_check, acronym_dict)
        return result_df[[column_to_check, 'phonetic_match']].sort_values(by='phonetic_match', ascending=False).head(top_n)
    
    elif method == 'levenshtein':
        result_df = levenshtein_match(user_input, customer_df, column_to_check, acronym_dict)
        return result_df[[column_to_check, 'levenshtein_score']].sort_values(by='levenshtein_score', ascending=False).head(top_n)
    
    elif method == "jarowinkler":
        result_df = jaro_winkler_match(user_input, customer_df, column_to_check, acronym_dict)
        return result_df[[column_to_check, 'jaro_winkler_score']].sort_values(by='jaro_winkler_score', ascending=False).head(top_n)
    
    elif method == "jaccard":
        result_df = jaccard_match(user_input, customer_df, column_to_check, acronym_dict)
        return result_df[[column_to_check, 'jaccard_score']].sort_values(by='jaccard_score', ascending=False).head(top_n)
    
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
    user_input = "John Sm Plumbing"
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='hybrid')
    print(f"Top {len(top_matches)} hybrid matches for '{user_input}':")
    print(top_matches)

    # Test with Levenshtein method
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='levenshtein')
    print(f"\nTop {len(top_matches)} Levenshtein matches for '{user_input}':")
    print(top_matches)

    # Test with n-gram method
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='ngram')
    print(f"\nTop {len(top_matches)} n-gram matches for '{user_input}':")
    print(top_matches)
    
    # Test with phonetic method
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='phonetic')
    print(f"\nTop {len(top_matches)} phonetic matches for '{user_input}':")
    print(top_matches)
    
    # Test with jarowinkler method
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='jarowinkler')
    print(f"\nTop {len(top_matches)} jarowinkler matches for '{user_input}':")
    print(top_matches)
    
    # Test with jaccard method
    top_matches = find_top_matches(user_input, df, 'full_name', acronym_dict=acronym_dict, method='jaccard')
    print(f"\nTop {len(top_matches)} jaccard matches for '{user_input}':")
    print(top_matches)