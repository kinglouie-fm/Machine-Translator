import matplotlib.pyplot as plt
import re
import bidict

def plot(data):
    result_bleu = []
    result_perp = []
    checkpoints = []

    for el in data:
         bleu = el['bleu']
         perp = el['perp']
         checkpoint = el['name']
         result_bleu.append(bleu)
         result_perp.append(perp)
         checkpoints.append(checkpoint)
    print(data)

    fig, axs = plt.subplots(2)
    axs[0].set(xlabel='Checkpoints', ylabel='BLEU')
    axs[0].plot(checkpoints, result_bleu)
    axs[1].set(xlabel='Checkpoints', ylabel='Perplexity')
    axs[1].plot(checkpoints, result_perp)
    plt.show()
