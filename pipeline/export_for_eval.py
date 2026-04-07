#!/usr/bin/env python3
import csv
from pathlib import Path
import sys

# ensure `src` is importable and import transliterate
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lao2ipa import transliterate

INPUT = 'lao_ipa_cleaned.csv'
OUT_SRC = 'src.txt'
OUT_REF = 'ref.txt'
OUT_EPITRAN = 'epitran.txt'
OUT_LAO2IPA = 'lao2ipa.txt'

# try to load epitran
try:
    import epitran
    EPI = epitran.Epitran('lao-Laoo')
except Exception as e:
    EPI = None
    _epitran_error = e


def main():
    if EPI is None:
        raise RuntimeError(f"epitran failed to import: {_epitran_error}")
    with open(INPUT, newline='', encoding='utf-8') as inf, \
         open(OUT_SRC, 'w', encoding='utf-8') as srcf, \
         open(OUT_REF, 'w', encoding='utf-8') as reff, \
         open(OUT_EPITRAN, 'w', encoding='utf-8') as epif, \
         open(OUT_LAO2IPA, 'w', encoding='utf-8') as laof:
        rdr = csv.DictReader(inf)
        for row in rdr:
            orth = row.get('orth','')
            pron = row.get('pron_true_ipa','')
            # remove spaces from orth
            orth_nospace = orth.replace(' ', '')
            # src: orth word
            srcf.write(orth_nospace + '\n')
            # ref: pron_true_ipa
            reff.write(pron + '\n')
            # epitran: use epitran module
            epi_out = EPI.transliterate(orth_nospace)
            epif.write(epi_out + '\n')
            # lao2ipa: use project's transliterate function
            lao2 = transliterate(orth_nospace, output='ipa')
            laof.write(lao2 + '\n')

if __name__ == '__main__':
    main()
