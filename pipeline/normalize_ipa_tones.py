#!/usr/bin/env python3
import csv
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "lao_ipa_tones.csv"
OUTPUT_MAP = HERE / "lao_ipa_cleaned.csv"

# Map common combining diacritics to tone markers
DIACRITIC_TO_TONE = {
    '\u0301': '˧˥',
    '\u0300': '˩',
    '\u0304': '˧',
    '\u0306': '˩˧',
    '\u030f': '˧˩',
    '\u0311': '˥˨',
}

def normalize_with_tones(s):
    # Split on whitespace and process each token independently so tones
    # attached to individual vowels are preserved after replacements.
    parts = s.split()
    out_parts = []
    for p in parts:
        p_nfd = unicodedata.normalize('NFD', p)
        tone = ''
        cleaned_chars = []
        for ch in p_nfd:
            if unicodedata.combining(ch):
                if not tone:
                    tone = DIACRITIC_TO_TONE.get(ch, '')
            else:
                cleaned_chars.append(ch)
        cleaned = ''.join(cleaned_chars)
        cleaned = unicodedata.normalize('NFC', cleaned)
        cleaned = normalize_ipa(cleaned)
        if tone:
            # insert tone before trailing punctuation (e.g. ')', '.', ',', etc.), otherwise append
            m = re.match(r'^(.*?)([)\]\}\.,;:!\?…]*)$', cleaned)
            if m:
                core, punct = m.group(1), m.group(2)
                cleaned = f"{core}{tone}{punct}"
            else:
                cleaned = f"{cleaned}{tone}"
        out_parts.append(cleaned)
    return ' '.join(out_parts)

def normalize_ipa(s):
    # ordered replacements
    s = s.replace('ɨaː', 'ɯːə̯')
    s = s.replace('ɨː', 'ɯː')
    s = s.replace('ɨ', 'ɯ')
    s = s.replace('y', 'j')
    s = s.replace('c', 't͡ɕ')
    return s

def remove_tones(s):
    # remove tone markers (˧˥, ˩, ˧, ˩˧, ˧˩, ˥˨)
    return re.sub(r'[˩˧˥˨]', '', s)

def main():
    with open(INPUT, newline='', encoding='utf-8') as inf, \
         open(OUTPUT_MAP, 'w', newline='', encoding='utf-8') as mapf:
        rdr = csv.reader(inf)
        mw = csv.writer(mapf)
        next(rdr) # Skip header
        mw.writerow(["orth", "pron_no_tones", "pron_with_tones"])
        for row in rdr:
            if not row:
                continue
            pron_with_tones = row[1].replace(' ', '') if len(row) > 1 else ''
            # remove all spaces from the transliteration output
            pron_no_tones = remove_tones(pron_with_tones).replace(' ', '')
            # remove spaces from 
            orth_nospace = row[0].replace(' ', '') if len(row) > 0 else ''
            mw.writerow([orth_nospace, pron_no_tones, pron_with_tones])

if __name__ == '__main__':
    main()
