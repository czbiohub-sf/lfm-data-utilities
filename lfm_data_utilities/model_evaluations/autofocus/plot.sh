#! /usr/bin/bash

./autofocus_evaluation.py evaluation_dataset_files/human-sorted/avt-1104.yml ~/autofocus/ulc-malaria-autofocus/trained_models/decent-donkey-431/best.pth -N 32 --output-dir ~/misc/avt-1104-dd
./autofocus_evaluation.py evaluation_dataset_files/human-sorted/avt-2004.yml ~/autofocus/ulc-malaria-autofocus/trained_models/decent-donkey-431/best.pth -N 32 --output-dir ~/misc/avt-2004-dd
./autofocus_evaluation.py evaluation_dataset_files/human-sorted/opp-0805.yml ~/autofocus/ulc-malaria-autofocus/trained_models/decent-donkey-431/best.pth -N 32 --output-dir ~/misc/opp-0805-dd
./autofocus_evaluation.py evaluation_dataset_files/human-sorted/spirit-0305.yml ~/autofocus/ulc-malaria-autofocus/trained_models/decent-donkey-431/best.pth -N 32 --output-dir ~/misc/spirit-0305-dd
./autofocus_evaluation.py evaluation_dataset_files/human-sorted/spirit-1904.yml ~/autofocus/ulc-malaria-autofocus/trained_models/decent-donkey-431/best.pth -N 32 --output-dir ~/misc/spirit-1904-dd
