from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        s = set(text)
        s = sorted(s)

        stoi = {}
        itos = {}
        for i, c in enumerate(s):
            stoi[c] = i
            itos[i] = c
        return (stoi, itos) 
        

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        result = []
        for t in text:
            result.append(stoi[t])
        return result

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        result = ""
        for i in ids:
            result += itos[i]
        return result
