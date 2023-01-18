#!/bin/bash
echo "Calculating corpus statistics for file $(realpath $1) ..."

# Number of words (including punctuation)
WORD_COUNT=$(cat $1 | wc -w )
echo "Number of words (including e.g. punctuation) : $WORD_COUNT"

# Number of real words
REAL_WORD_COUNT=$(grep -o -E '\w+' $1 | tr '[A-Z]' '[a-z]' | wc -w)
echo "Number of real words                         : $REAL_WORD_COUNT"

# Number of different words
REAL_DIFFERENT_WORD_COUNT=$(grep -o -E '\w+' $1 | tr '[A-Z]' '[a-z]' | sort -u | wc -w )
echo "Number of different words                    : $REAL_DIFFERENT_WORD_COUNT"

# Average sentence length
LINES=$(cat $1 | wc -l )
AVERAGE_SENTENCE_LENGTH=$(($REAL_WORD_COUNT / $LINES))
echo "Average sentence length                      : $AVERAGE_SENTENCE_LENGTH"
