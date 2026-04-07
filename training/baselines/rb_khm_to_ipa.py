import argparse
from khmernltk import word_tokenize # https://pypi.org/project/khmer-nltk/
from tqdm import tqdm

# https://en.wikipedia.org/wiki/Help:IPA/Khmer 
outputs = {
    "ipa": 0,
}

# dictionary of Khmer consonants and assignments (IPA).
# This includes the n̪ diacritic (the bracket), should make one without to see how it works for downstream tasks
# khmer_consonant_assignments = {
#     "ក": [["k", "k", 1]],
#     "ខ": [["kʰ", "k", 1]],
#     "គ": [["k", "k", 2]],
#     "ឃ": [["kʰ", "k", 2]],
#     "ង": [["ŋ", "ŋ", 2]],
#     "ច": [["c", "c", 1]],
#     "ឆ": [["cʰ", "cʰ", 1]],
#     "ជ": [["c", "c", 2]],
#     "ឈ": [["cʰ", "cʰ", 2]],
#     "ញ": [["ɲ", "ɲ", 2]],
#     "ដ": [["ɗ", "t̪", 1]],
#     "ឋ": [["t̪ʰ", "t̪", 1]],
#     "ឌ": [["ɗ", "t̪", 2]],
#     "ឍ": [["t̪ʰ", "t̪", 2]],
#     "ណ": [["n̪", "n̪", 1]],
#     "ត": [["t̪", "t̪", 1]],
#     "ថ": [["t̪ʰ", "t̪", 1]],
#     "ទ": [["t̪", "t̪", 2]],
#     "ធ": [["t̪ʰ", "t̪", 2]],
#     "ន": [["n̪", "n̪", 2]],
#     "ប": [["ɓ", "p", 1]],
#     "ផ": [["pʰ", "p", 1]],
#     "ព": [["p", "p", 2]],
#     "ភ": [["pʰ", "p", 2]],
#     "ម": [["m", "m", 2]],
#     "យ": [["j", "j", 2]],
#     "រ": [["r", "", 2]],
#     "ល": [["l̪", "l̪", 2]],
#     "វ": [["ʋ", "ʋ", 2]],
#     "ស": [["s", "h", 1]],
#     "ហ": [["h", "h", 1]],
#     "ឡ": [["l̪", "l̪", 1]],
#     "អ": [["ʔ", "ʔ", 1]],
# }

# Without the n̪ diacritic (the bracket)
khmer_consonant_assignments = {
    "ក": [["k", "k", 1]],
    "ខ": [["kʰ", "k", 1]],
    "គ": [["k", "k", 2]],
    "ឃ": [["kʰ", "k", 2]],
    "ង": [["ŋ", "ŋ", 2]],
    "ច": [["c", "c", 1]],
    "ឆ": [["cʰ", "cʰ", 1]],
    "ជ": [["c", "c", 2]],
    "ឈ": [["cʰ", "cʰ", 2]],
    "ញ": [["ɲ", "ɲ", 2]],
    "ដ": [["ɗ", "t", 1]],
    "ឋ": [["tʰ", "t", 1]],
    "ឌ": [["ɗ", "t", 2]],
    "ឍ": [["tʰ", "t", 2]],
    "ណ": [["n", "n", 1]],
    "ត": [["t", "t", 1]],
    "ថ": [["tʰ", "t", 1]],
    "ទ": [["t", "t", 2]],
    "ធ": [["tʰ", "t", 2]],
    "ន": [["n", "n", 2]],
    "ប": [["ɓ", "p", 1]],
    "ផ": [["pʰ", "p", 1]],
    "ព": [["p", "p", 2]],
    "ភ": [["pʰ", "p", 2]],
    "ម": [["m", "m", 2]],
    "យ": [["j", "j", 2]],
    "រ": [["r", "", 2]],
    "ល": [["l", "l", 2]],
    "វ": [["ʋ", "ʋ", 2]],
    "ស": [["s", "h", 1]],
    "ហ": [["h", "h", 1]],
    "ឡ": [["l", "l", 1]],
    "អ": [["ʔ", "ʔ", 1]],
}

khmer_vowel_assignments = {
    "ា": [["aa", "eə"]],
    "ិ": [["e", "i"]],
    "ី": [["ej", "ii"]],
    "ឹ": [["ɜ", "ɨ"]],
    "ឺ": [["ə", "ɨ"]],
    "ុ": [["o", "u"]],
    "ូ": [["oo", "uu"]],
    "ួ": [["uə", "uə"]],
    # "ូវ": [["ɜw", "ɨw"]],
    "ើ": [["aə", "ə"]],
    "ឿ": [["ɨə", "ɨə"]],
    "ៀ": [["iə", "iə"]],
    "េ": [["e", "ee"]],
    "ែ": [["ae", "ɛɛ"]],
    "ៃ": [["aj", "ɪj"]],
    "ោ": [["ao", "oo"]],
    "ៅ": [["aw", "əw"]],
    "ុំ": [["om", "um"]],
    "ំ": [["ɑm", "um"]],
    "ាំ": [["am", "oə̯m"]],
    "ះ": [["ah", "eəh"]],
    "ិះ": [["eh", "ih"]],
    "ឹះ": [["ɜh", "ɨh"]],
    "ែះ": [["aeh", "eh"]],
    "ុះ": [["oh", "uh"]],
    "េះ": [["eh", "ih"]],
    "ោះ": [["ɑh", "uəh"]],
    "័": [["e", "e"]],
}

khmer_numbers = {
    "០": "0",
    "១": "1",
    "២": "2",
    "៣": "3",
    "៤": "4",
    "៥": "5",
    "៦": "6",
    "៧": "7",
    "៨": "8",
    "៩": "9",
}

khmer_diacritics = {
    "់",
    "៏",
    "័",
    "៉",
}

other_conversions = {
    "ឥ": "ʔe", # Independent vowel
    "ឦ": "ʔej", # Independent vowel
    "ឧ": "ʔu", # Independent vowel
    "ឩ": "ʔu", # Independent vowel
    "ឪ": "ʔɜw", # Independent vowel
    "ឫ": "rɨ", # Independent vowel
    "ឬ": "rɨɨ", # Independent vowel
    "ឭ": "lɨ", # Independent vowel
    "ឮ": "lɨɨ", # Independent vowel
    "ឯ": "ʔae",  # Independent vowel
    "ឰ": "ʔaj", # Independent vowel
    "ឱ": "ʔao", # Independent vowel
    "ឲ": "ʔao", # Independent vowel
    "ឳ": "ʔaw", # Independent vowel
    "។": ".",  # Khmer period
    "៕": ".",  # Khmer full stop
    "៚": ".",  # Khmer paragraph separator
    "៖": ":",  # Khmer colon
    "«": "\"", # Khmer left double quotation mark
    "»": "\"", # Khmer right double quotation mark
}

khmer_characters = set.union(set(khmer_consonant_assignments.keys()), set(khmer_vowel_assignments.keys()))


def khmer2ipa(khmer_source: str, output: str = 'ipa'):
    if khmer_source == '':
        return ''

    # Segment into words
    words = word_tokenize(khmer_source, return_tokens=True)

    # print("Words: ", words)

    transliterated = []

    for word in words:
        # Convert string to list
        khmer_array = list(word)
        # print("Converted Khmer source to array: ", khmer_array)

        # Merge consonants
        khmer_array = merge_consonants(khmer_array)
        # print("Merged consonants: ", khmer_array)

        # Merge vowels
        khmer_array = merge_vowels(khmer_array)
        # print("Merged vowels: ", khmer_array)

        # Transliterate
        khmer_array = transliterate(khmer_array)

        transliterated.append(khmer_array)

    return "".join(transliterated)

def merge_consonants(khmer_array: list):
    """
    Merge consants when there is the subconsonant character between ្
    """
    khmer_consonants = set(khmer_consonant_assignments.keys())
    max_len = len(khmer_array)
    pos = 0
    result = []

    while pos < max_len:
        # Check for the longer pattern: Consonant + '្' + Consonant + '្' + Consonant
        # Example: ក្ខ្គ (K + COENG + KH + COENG + K)
        if (pos + 4 < max_len and
            khmer_array[pos] in khmer_consonants and
            khmer_array[pos + 1] == "្" and
            khmer_array[pos + 2] in khmer_consonants and
            khmer_array[pos + 3] == "្" and
            khmer_array[pos + 4] in khmer_consonants):
            # Merge the three consonants and append to the result
            merged_chars = (
                khmer_array[pos] +
                khmer_array[pos + 2] +
                khmer_array[pos + 4]
            )
            result.append(merged_chars)
            pos += 5  # Advance position past the 5 characters processed
        # Check for the shorter pattern: Consonant + '្' + Consonant
        # Example: ក្ខ (K + COENG + KH)
        elif (pos + 2 < max_len and
              khmer_array[pos] in khmer_consonants and
              khmer_array[pos + 1] == "្" and
              khmer_array[pos + 2] in khmer_consonants):
            # Merge the two consonants and append to the result
            merged_chars = (
                khmer_array[pos] +
                khmer_array[pos + 2]
            )
            result.append(merged_chars)
            pos += 3  # Advance position past the 3 characters processed
        else:
            # No merging pattern found, append the current character as is
            result.append(khmer_array[pos])
            pos += 1  # Advance position by 1

    return result

def merge_vowels(khmer_array: list):
    """
    Merge consecutive vowels.
    """
    khmer_vowels = set(khmer_vowel_assignments.keys())
    max_len = len(khmer_array)
    pos = 0
    result = []

    while pos < max_len:
        cur_char = khmer_array[pos]

        # Check if the current character is a vowel
        if cur_char in khmer_vowels:
            # If it's a vowel, start merging.
            merged_vowel_sequence = cur_char
            temp_pos = pos + 1

            # Continue merging as long as subsequent characters are also vowels.
            while temp_pos < max_len and khmer_array[temp_pos] in khmer_vowels:
                merged_vowel_sequence += khmer_array[temp_pos]
                temp_pos += 1

            # Add the complete merged vowel sequence to the result.
            result.append(merged_vowel_sequence)
            # Advance the main position to after the merged sequence.
            pos = temp_pos
        else:
            # If it's not a vowel, add it as is.
            result.append(cur_char)
            pos += 1

    return result

def determine_vowel_type(consonants: list):
    """
    Determines the vowel type based on a heuristic of strong or weak consonants.
    Prioritizes strong consonants; falls back to weak if no strong consonants are found.
    Default to type 1 if no relevant consonants are found.
    """
    weak_consonants_set = {"ង", "ញ", "ណ", "ន", "ម", "រ", "ល", "វ", "ហ"}

    # Flatten the list of consonant clusters into a single list of individual consonants
    all_individual_consonants = [c for cluster in consonants for c in cluster]

    # Filter out weak consonants to get strong consonants
    strong_consonants = [
        c for c in all_individual_consonants
        if c in khmer_consonant_assignments and c not in weak_consonants_set
    ]

    vowel_types = []

    if strong_consonants:
        # Use strong consonants if available
        for consonant in strong_consonants:
            vowel_types.append(khmer_consonant_assignments[consonant][0][2])
    else:
        # Fallback to weak consonants if no strong consonants are found
        weak_consonants = [
            c for c in all_individual_consonants
            if c in khmer_consonant_assignments and c in weak_consonants_set
        ]
        for consonant in weak_consonants:
            vowel_types.append(khmer_consonant_assignments[consonant][0][2])

    if not vowel_types:
        return 1  # Default to type 1 if no relevant consonants were found

    # Return the most common vowel type
    return max(set(vowel_types), key=vowel_types.count)
    

def transliterate(khmer_array: list, output: str = "ipa"):
    """
    Treat consonant clusters at the end of the array as a final (the first of the sub-list) and
    all others as initials.
    Figure out the weak consonant in the list, use that to determine vowel sounds.
    If there are two consonant clusters in a row, add the default vowel sound, otherwise, use the vowel after it.
    Ignore diacritic rules for now.
    If it ends on a vowel, that's fine.
    """
    output_index = outputs[output]
    default_vowel_sounds = ['ɑɑ', 'ɔɔ' ]

    # Determine the vowel type (1 or 2)
    consonants = [c for c in khmer_array if c not in set(khmer_vowel_assignments.keys())]
    vowel_type = determine_vowel_type(consonants) - 1
    # print(f"Vowel type: {vowel_type}") # Debug

    # Transliterate
    transliterated = []

    # If the list is a single consonant, return the transliteration directly
    if len(khmer_array) == 1 and khmer_array[0] in khmer_consonant_assignments:
        return khmer_consonant_assignments[khmer_array[0]][output_index][0] + default_vowel_sounds[vowel_type]

    # Add the default vowel sounds to the first consonant when there are two consonants in a row
    for i in range(len(khmer_array) - 2, -1, -1): # Iterate backwards
        if (khmer_array[i][0] in khmer_consonant_assignments and
            khmer_array[i + 1][0] in khmer_consonant_assignments):
            # Insert the default vowel sound after the first consonant
            khmer_array.insert(i + 1, default_vowel_sounds[vowel_type])

    # Five categories: consonant cluster, vowel cluster, numerals, diacritics, and anything else.
    for i, cluster in enumerate(khmer_array):
        # print(f"Cluster {i}: {cluster}") # Debug

        # Handle finals
        if i == len(khmer_array) - 1 and cluster[0] in set(khmer_consonant_assignments.keys()):
            # Use the the final sound of cluter[0] and ignore the rest.
            transliterated.append(khmer_consonant_assignments[cluster[0]][output_index][1])

        # Consonant clusters with more than one consonant
        elif len(cluster) > 1 and cluster[0] in set(khmer_consonant_assignments.keys()):
            temp = ''
            for i in range(len(cluster)):
                temp += khmer_consonant_assignments[cluster[i]][output_index][0]
            transliterated.append(temp)

        # Single characters
        elif cluster in khmer_consonant_assignments:
            # Convert consonants
            transliterated.append(khmer_consonant_assignments[cluster][output_index][0])
        elif cluster in khmer_vowel_assignments:
            # Convert vowels
            transliterated.append(khmer_vowel_assignments[cluster][output_index][vowel_type])
        
        # Compound vowels
        elif len(cluster) > 1 and cluster[0] in set(khmer_vowel_assignments.keys()):
            # Convert compound vowels not included in the predefined list
            temp = ''
            for i in range(len(cluster)):
                temp += khmer_vowel_assignments[cluster[i]][output_index][vowel_type]
            transliterated.append(temp)
        
        elif cluster in khmer_numbers:
            # Convert numbers
            transliterated.append(khmer_numbers[cluster])
        elif cluster in khmer_diacritics:
            # Ignore diacritics for now
            if cluster in khmer_diacritics:
                continue
        elif cluster in other_conversions:
            # Convert other characters
            transliterated.append(other_conversions[cluster])
        else:
            # Anything else, just append it as is
            transliterated.append(cluster)
    
    return "".join(transliterated)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Khmer text to IPA.")
    parser.add_argument("input_file", type=str, help="Path to the input Khmer text file.")
    parser.add_argument("output_file", type=str, help="Path to the output IPA text file.")
    args = parser.parse_args()

    print("Starting Khmer to IPA transliteration...")

    with open(args.input_file, "r") as f:
        lines = f.readlines()

    with open(args.output_file, "w") as out_f:
        for line in tqdm(lines):
            line = line.strip()
            if line:
                ipa = khmer2ipa(line, output='ipa')
                out_f.write(ipa + "\n")
            else:
                out_f.write("\n")
                
    print(f"Transliteration complete. IPA output saved to {args.output_file}.")