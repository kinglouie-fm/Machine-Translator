import levenshtein
import bleu
import argparse


# Calculates the WER, Word Error Rate, on a sentence level
def sentence_wer(ref, hyp):
    return levenshtein.levenshtein_distance(ref, hyp) / len(ref.split())


# Calculates the PER, Position-independent Error Rate, on a sentence level
def sentence_per(ref, hyp):
    # Building sets from the two given sentences
    ref_set = set(ref.split())
    hyp_set = set(hyp.split())

    # Calculates the lengths of the two given sentences
    ref_len = len(ref.split())
    hyp_len = len(hyp.split())

    # Calculates how many words occur in both sentences, disregarding double occurrences or the order
    matches = len(ref_set.intersection(hyp_set))

    # Calculates the PER
    per = 1 - ((matches - (max(0, hyp_len - ref_len))) / ref_len)

    # print("%s matches, hyp_len: %s, ref_len %s, PER: %s" % (matches, hyp_len, ref_len, per))
    return per


# Calculates the WER, Word Error Rate, for a list of pairs of hypotheses and references
def calc_wer(big_l):
    total_wer = 0  # The sum of all PER.
    for i in range(0, len(big_l) - 1):
        ref_i_list = big_l[i][0]
        hyp_i_list = big_l[i][1]
        ref_i = ' '.join(ref_i_list)
        hyp_i = ' '.join(hyp_i_list)

        # Adds the PER of a sentence to the total PER
        total_wer += sentence_wer(ref_i, hyp_i)

    average_wer = total_wer / (len(big_l) - 1)
    return average_wer


# Calculates the PER, Position-independent Error Rate, for a list of pairs of hypotheses and references
def calc_per(big_l):
    total_per = 0  # The sum of all PER.
    for i in range(0, len(big_l) - 1):
        ref_i_list = big_l[i][0]
        hyp_i_list = big_l[i][1]
        ref_i = ' '.join(ref_i_list)
        hyp_i = ' '.join(hyp_i_list)

        # Adds the PER of a sentence to the total PER
        total_per += sentence_per(ref_i, hyp_i)

    average_per = total_per / (len(big_l) - 1)
    return average_per


# Calculates the BLEU score for a list of hypotheses and references
def calc_bleu(hypotheses, references):
    return bleu.bleu(hypotheses, references)


# Calculates all three metrics for a reference and a hypothesis text file
def calc_metrics(ref, hyp):
    # Open both files
    reference = open(ref, 'r', encoding="utf-8")
    hypothesis = open(hyp, 'r', encoding="utf-8")

    # Converts both texts to lists of the sentences
    reference_text = reference.read().split('\n')
    hypothesis_text = hypothesis.read().split('\n')

    # Close both files again
    reference.close()
    hypothesis.close()

    # Build lists of hypotheses and references
    references = []
    hypotheses = []
    for i in range(0, len(reference_text) - 1):
        references.append(reference_text[i].split())
        hypotheses.append(hypothesis_text[i].split())
        
    # Calc and print the all the metrics
    print(f"Calculating metrics for reference '{ref}' and hypothesis '{hyp}' ...")
    # wer_score = calc_wer(big_l)
    # per_score = calc_per(big_l)
    bleu_score = calc_bleu(hypotheses, references)
    # print(f"Word Error Rate                 (0 is best) : {wer_score}")
    # print(f"Position-independent Error Rate (0 is best) : {per_score}")
    print(f"BLEU score                      (1 is best) : {bleu_score} \n")


# Calculate all scores for all three hypothesis files
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="(relative) path of the reference file")
    parser.add_argument("hypothesis", help="(relative) path of the hypothesis file")
    args = parser.parse_args()
    calc_metrics(args.reference, args.hypothesis)

    # Calculating the metrics clearly shows that the hypothesis in file 'newstest.hyp3' is by far the worst. Hyp1 and hyp2 only show small differences, however hyp1 is the best of all three.yx
