import math
import sys

# n: n (integer)
# hypothesis: hypothesis (array of strings)
# reference: reference (array of strings)
def ngram_matches(n, hypothesis, reference):
    h_ngrams = [] # array of array of strings
    r_ngrams = [] # array of array of strings

    # Construct ngrams
    # Construct set of ngrams for hypothesis
    for i in range(len(hypothesis) - n +1):
        h_ngrams.append([])
        for j in range(n):
            h_ngrams[i].append(hypothesis[i+j])
    
    # Construct set of ngrams for reference
    for i in range(len(reference) - n +1):
        r_ngrams.append([])
        for j in range(n):
            r_ngrams[i].append(reference[i+j])
    
    #print(h_ngrams)
    #print(r_ngrams)
    #print("----------------------------")

    # Check for matches between the two sets of ngrams
    matches = 0

    for h in h_ngrams:
        for r in r_ngrams:
            if h == r:
                matches += 1
                r_ngrams.remove(r) # Removes the first occurence of the r_ngram with value r and not necessarily this r but in our case that is irrelevant
                break
                #print(r_ngrams)

    #print("----------------------------")
    #print(h_ngrams)
    #print(r_ngrams)
    #print(f"Matches: {matches}")
    return matches


# n: n (integer)
def ngram_precision(n, hypotheses, references):
    numerator = 0
    denominator = 0

    for hypothesis, reference in zip(hypotheses, references):
        if(len(hypothesis) >= n):
            numerator += ngram_matches(n, hypothesis, reference)
            denominator += (len(hypothesis) - n +1)

    if (denominator == 0):
        precision = float(-1)
    else:
        precision = numerator / denominator

    # print(f"{n}-gram precision : {precision}")
    return precision


# c: length of hypothesis (integer)
# r: length of reference (integer)
def brevity_penalty(c,r):
    if c > r:
        return 1
    else:
        return math.e**(1-(r/c))


# L: list of hypotheses and list of references (each arrays of arrays of strings)
def bleu(hypotheses, references):
    max_ngrams = 4

    bp = brevity_penalty(len(hypotheses), len(references))
    
    # calculate accumulated precision
    ap = 0
    for n in range(max_ngrams):
        current_ngram_precision = ngram_precision(n+1, hypotheses, references)
        if current_ngram_precision == -1:
            max_ngrams = n-1
            break

        elif current_ngram_precision == 0:
            return 0.0 # shortcut for adding math.log(SUPER_SMALL_NUMBER) which effectively leads to a BLEU score close to 0

        else:
            ap += math.log(current_ngram_precision)

    ap = math.e**((1/max_ngrams)*ap)

    bleu_score = bp * ap
    # print(f"Bleu score       : {bleu_score}")
    return bleu_score