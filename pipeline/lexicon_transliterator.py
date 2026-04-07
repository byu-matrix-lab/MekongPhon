class TrieNode:
    def __init__(self):
        self.children = {}
        self.value = None  # phonetic if terminal

class GreedyTransliterator:
    def __init__(self, mapping):
        self.root = TrieNode()
        for orth, pron in mapping.items():
            node = self.root
            for ch in orth:
                node = node.children.setdefault(ch, TrieNode())
            node.value = pron

    def transliterate(self, text):
        i = 0
        output = []
        while i < len(text):
            node = self.root
            longest_match = None
            longest_value = None
            j = i
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.value:
                    longest_match = j
                    longest_value = node.value
            if longest_match:
                output.append(longest_value)
                i = longest_match
            else:
                output.append(text[i])  # no match, copy character
                i += 1
        return "".join(output)

if __name__ == "__main__":
    # Usage
    pairs = {
        "បាន": "baan",
        "ការ": "kaa",
        "មាន": "mien",
        "ជា": "cie",
        "នៅ": "nɨv",
        "ដែល": "dael",
        "ក្នុង": "knoŋ",
        "នេះ": "nih",
        "ថា": "tʰaa",
        "ទៅបាន": "tɨvbaan",
        "ទៅ": "tɨv",
        "ពី": "pii",
        "មិន": "mɨn",
    }

    trans = GreedyTransliterator(pairs)
    print(trans.transliterate("ទៅបានទៅក្នុងនេះ"))
