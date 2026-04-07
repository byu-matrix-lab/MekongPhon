import requests
import urllib.parse
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse
import csv
import time

import requests
import urllib.parse
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse
import csv
import time

# Function for returning the response after requesting a single word in either khmer or lao (or thai)
def get_sealang_data(word, language="khmer"):
    if language not in ["khmer", "lao"]:
        raise ValueError("Language must be 'khmer' or 'lao'.")

    if language == "lao":
        base_url = "http://sealang.net/lao/search.pl"
        params = {
            "dict": "lao",
            "hasFocus": "orth",
            "approx": "",
            "orth": word,
            "phone": "",
            "def": "",
            "anon": "on",
            "matchEntry": "any",
            "matchLength": "word",
            "matchPosition": "any",
            "source": "",
            "ety": "",
            "pos": "",
            "usage": "",
            "subject": "",
            "useTags": "1"
        }
    else:
        base_url = "http://sealang.net/khmer/search.pl"
        params = {
            "dict": "khmer",
            "hasFocus": "orth",
            "approx": "",
            "orth": word,
            "phone": "",
            "def": "",
            "anon": "on",
            "matchEntry": "any",
            "matchLength": "word",
            "matchPosition": "any",
            "source": "headley77",
            "ety": "",
            "pos": "",
            "usage": "",
            "subject": "",
            "useTags": "1"
        }

    encoded_params = urllib.parse.urlencode(params, safe=':/')
    url = f"{base_url}?{encoded_params}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as e:
        if response.status_code == 500:
            print(f"⚠️  Server error (500) for word '{word}' — skipping.")
        else:
            print(f"⚠️  HTTP error {response.status_code} for '{word}': {e}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Request failed for '{word}': {e}")
    return None


def parse_sealang_html_file_lao(html):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for entry_tag in soup.find_all("entry"):
        formx_tag = entry_tag.find("formx")
        if formx_tag and formx_tag.get('id', '').startswith("kerr72"):
            orth = entry_tag.get("orthtarget")
            pron_no_tones = entry_tag.get("prontarget")
            pron_with_tones = formx_tag.find("pron").text if formx_tag.find("pron") else ""
            return True, [{"orth": orth, "pron_no_tones": pron_no_tones, "pron_with_tones": pron_with_tones}]
    return False, []


def parse_sealang_html_file_khmer(html):
    soup = BeautifulSoup(html, "html.parser")
    for entry_tag in soup.find_all("entry"):
        orth = entry_tag.get("orthtarget")
        pron_no_tones = entry_tag.get("prontarget")
        return True, [{"orth": orth, "pron_no_tones": pron_no_tones, "pron_with_tones": ""}]
    return False, []


def main():
    parser = argparse.ArgumentParser(description="Scrape Sealang dictionary data for a list of words.")
    parser.add_argument("language", choices=["khmer", "lao"], help="Language to scrape (khmer or lao)")
    parser.add_argument("input_file", help="Path to input text file with list of words (one per line)")
    parser.add_argument("output_file", help="Path to output CSV file")
    args = parser.parse_args()

    language = args.language
    input_file = args.input_file
    output_file = args.output_file

    with open(input_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    failed_words = []

    with open(output_file, "a", encoding="utf-8", newline='') as csvfile:
        fieldnames = ["orth", "pron_no_tones", "pron_with_tones"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for word in tqdm(words, desc="Processing words"):
            html = get_sealang_data(word, language=language)
            if not html:
                failed_words.append(word)
                continue  # Skip failed requests

            if language == "khmer":
                found, results = parse_sealang_html_file_khmer(html)
            else:
                found, results = parse_sealang_html_file_lao(html)

            if found:
                for res in results:
                    writer.writerow(res)
            else:
                failed_words.append(word)

            # Randomized delay
            time.sleep(1 + (2 * (0.5 - time.time() % 1)))

    if failed_words:
        with open(language + "_failed_words.txt", "a", encoding="utf-8") as f:
            for w in failed_words:
                f.write(w + "\n")

def test_lao():
    # Test lao
    lao_word = "ແລະ"
    lao_html = get_sealang_data(lao_word, language="lao")
    found, lao_results = parse_sealang_html_file(lao_html)
    print(f"Lao word '{lao_word}' found: {found}")
    print(lao_results)

def test_khmer():
    # Test khmer
    khmer_word = "ការ"
    khmer_html = get_sealang_data(khmer_word, language="khmer")
    found, khmer_results = parse_sealang_html_file(khmer_html)
    print(f"Khmer word '{khmer_word}' found: {found}")
    print(khmer_results)


if __name__ == "__main__":
    main()


