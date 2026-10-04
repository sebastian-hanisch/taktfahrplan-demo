"""SplitMix64 (Vigna): kleiner, gut gemischter 64-Bit-Zufallsgenerator in reiner Ganzzahl-Arithmetik. numpy garantiert keine
versionsstabilen Zufallsströme, die CI installiert die neueste Version; SplitMix64 liefert überall dieselbe Zahlenfolge."""

_MASK = (1 << 64) - 1


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik (Muster au_scenario.py)."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n):
        """Ganzzahl in 0..n-1 (Modulo-Verzerrung bei n <= 1000 liegt unter 1e-16)."""
        return self.next() % n

    def randrange(self, n):
        return self.below(n)
