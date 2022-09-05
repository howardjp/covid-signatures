#!/usr/bin/env bash

DATADIR="./data/sepsis/preprocessed"
OUTPUT="./data/sepsis/merged.csv"

rm "$OUTPUT"; touch $OUTPUT
for i in "$DATADIR"/*; do
  echo "processing file" "$i"
  pname=$(basename "$i" .csv)
  sed "s/^/$pname,/" < "$i" >> $OUTPUT
done
