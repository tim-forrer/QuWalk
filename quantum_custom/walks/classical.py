from scipy.special import comb
from quantum_custom.core import PlotData

def prob(m, N):
    """
    Given target position m and N flips of the coin, returns the probability of finishing on m.

    Using convention that heads is +1 and tails is -1.
    Assuming m > 0 we know that we need at least m heads, and following that equal numbers of heads and tails.
    Then count the number of combinations that are possible of having that number of heads and divide by total number of combinations.

    Since this distribution is symmetric, if m < 0 we can reverse its sign as the probability will be the same.
    """
    if m < 0:
        m = -m
    Nheads = int(m + (N - m) / 2)

    prob = comb(N, Nheads, exact = False) * (0.5)**N
    return prob

def pdf(x, N):
    """
    The probability distribution function of a classical walk after N coin flips.

    Returns an PlotData() instance.
    """
    probs = []
    for x_val in x:
        probs.append(prob(x_val, N))
    start_index = N % 2
    x = x[start_index::2]
    probs = probs[start_index::2]
    return PlotData(x, probs, N)
