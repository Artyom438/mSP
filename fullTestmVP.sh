#!/bin/sh
for item in "/home/artyom/ScientificWork/mRNA localization/UMSLP/stability/features/"test_*.csv; do
    outPath=predictions/"${item##*/}"
    conda run -n keras2 python3 model.py "$item" "$outPath" mVP
done
