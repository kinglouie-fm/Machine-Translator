# Calculates the levenshtein_distance of two sentences (strings).
def levenshtein_array(ref, hyp):
    hyp_len = len(hyp.split())
    ref_len = len(ref.split())
    ls_arr = [[-1 for _ in range(hyp_len + 1)] for _ in range(ref_len + 1)]  # Declaration of the levenshtein array

    for i in range(ref_len + 1):
        for j in range(hyp_len + 1):
            if i == 0 and j == 0:
                ls_arr[i][j] = 0  # Value at [0][0]
            elif i == 0:
                ls_arr[i][j] = ls_arr[i][j - 1] + 1  # Values in the first row
            elif j == 0:
                ls_arr[i][j] = ls_arr[i - 1][j] + 1  # Values in the first column

            else:  # i > 0 AND j > 0
                if hyp.split()[j - 1] == ref.split()[i - 1]:  # If there is a match between two words
                    ls_arr[i][j] = ls_arr[i - 1][j - 1]
                else:  # If there is no match between the two words, choose the easiest way, replace, insert, delete
                    ls_arr[i][j] = min(ls_arr[i - 1][j - 1], ls_arr[i][j - 1], ls_arr[i - 1][j]) + 1

    # Prints the array in a proper way.
    # for i in range(ref_len + 1):
    #     print(ls_arr[i])
    return ls_arr


def levenshtein_distance(ref, hyp):
    # Returns the levenshtein_distance, the value in the lower right corner
    return levenshtein_array(ref, hyp)[len(ref.split())][len(hyp.split())]


def levenshtein_operations(ref, hyp):
    ls_arr = levenshtein_array(ref, hyp)
    edit_arr = []  # List of operations, from end to beginning! Printed from back at the end

    i = len(ref.split())
    j = len(hyp.split())

    while i != 0 or j != 0:
        if i == 0:  # upper row
            edit_arr.append("delete " + hyp.split()[j - 1])
            j -= 1
        elif j == 0:  # left most column
            edit_arr.append("insert " + ref.split()[i - 1])
            i -= 1

        else:
            upper_val = ls_arr[i - 1][j]  # the upper value of the current i, j.
            left_val = ls_arr[i][j - 1]  # the left value of the current i, j.
            upper_left_val = ls_arr[i - 1][j - 1]  # the upper left value of the current i, j.

            if upper_val == ls_arr[i][j] - 1:  # Insertion
                edit_arr.append("insert " + ref.split()[i - 1])
                i -= 1
            elif upper_left_val == ls_arr[i][j] - 1:  # Substitution
                edit_arr.append("replace " + hyp.split()[j - 1] + " by " + ref.split()[i - 1])
                i -= 1
                j -= 1
            elif left_val == ls_arr[i][j] - 1:  # Deletion
                edit_arr.append("delete " + hyp.split()[j - 1])
                j -= 1
            elif upper_left_val == ls_arr[i][j]:  # Match!
                edit_arr.append("match of " + ref.split()[i - 1])
                i -= 1
                j -= 1

            else:  # Should NEVER occur
                print("The value at i = " + str(i) + " and j = " + str(j), end="")
                print("does not follow the rules of the levenshtein dist.")
                break

    output = ""
    for k in range(1, len(edit_arr) + 1):  # Prints array from the back.
        output += "%d. %s \n" % (k, edit_arr[-k])
    output += "Levenshtein-distance: " + str(ls_arr[len(ref.split())][len(hyp.split())])
    return output


if __name__ == '__main__':
    print(levenshtein_operations("B A N A N E N", "S P A N A N I E N"))
