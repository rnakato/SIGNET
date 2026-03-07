# Epinion analysis

wget -P data http://snap.stanford.edu/data/soc-sign-epinions.txt.gz

signet --format snap \
       --snap data/soc-sign-epinions.txt.gz \
       --method leiden-mod-alpha \
       --alpha 0.6 \
       --resolution 1.0 \
       --seed 12345 \
       --out-prefix results/epinions_alpha06

exit

# For TSV files
signet --pos data/hsc_CDI_score_data_thre10.txt \
  --neg data/hsc_EEI_score_data_thre5.txt \
  --thre-pos 10 --thre-neg 5 --method leiden-mod-alpha \
  --alpha 0.6 --resolution 1.0 --out partition.tsv

# For MAT files
signet --format mat --mat data/epinions_data.mat --method leiden-mod-alpha --alpha 0.6 --resolution 1.0
