#\!/bin/bash
PYTHON=/u/erdos/csga/ntrinhvanminh/.conda/envs/vprtutorial/bin/python
cd ~/vpr-research/VPR_Tutorial
echo "Starting full Nordland experiments on $(hostname) at $(date)"
echo ">>> STEP 1: Discovery-then-Recognition"
$PYTHON experiments/discovery_then_recognition.py --data_dir images/Nordland_filtered --ref_condition summer --query_condition winter --max_images 0 --img_ext "*.png" 2>&1 | tee results/nordland_full_discovery.log
echo ">>> STEP 2: Threshold comparison with Vysotska"
$PYTHON experiments/compare_thresholds.py --data_dir images/Nordland_filtered --ref_condition summer --query_condition winter --max_images 0 --img_ext "*.png" 2>&1 | tee results/nordland_full_comparison.log
echo "All experiments DONE at $(date)"
