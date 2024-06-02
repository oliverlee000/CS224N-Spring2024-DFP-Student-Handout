echo "Trying out different numbers of linear layers"
echo "Testing sst"
python3 multitask_classifier.py --use_gpu --epochs 6 --fine-tune-mode last-linear-layer --cos_sim_loss y --neg_rankings_loss y --task sst --sst_layers 2 --sst_hidden_size 100 > sst_test_1.txt
python3 multitask_classifier.py --use_gpu --epochs 6 --fine-tune-mode last-linear-layer  --cos_sim_loss y --neg_rankings_loss y --task sst --sst_layers 3 --sst_hidden_size 100 > sst_test_2.txt
python3 multitask_classifier.py --use_gpu --epochs 6 --fine-tune-mode last-linear-layer  --cos_sim_loss y --neg_rankings_loss y --task sst --sst_layers 4 --sst_hidden_size 100 > sst_test_3.txt
echo "Testing para"
python3 multitask_classifier.py --use_gpu --epochs 6 --fine-tune-mode last-linear-layer --cos_sim_loss y --neg_rankings_loss y --task para --para_layers 0 --para_hidden_size 40 > para_test_1.txt
python3 multitask_classifier.py --use_gpu --epochs 6 --fine-tune-mode last-linear-layer  --cos_sim_loss y --neg_rankings_loss y --task para --para_layers 1 --para_hidden_size 40 > para_test_2.txt
python3 multitask_classifier.py --use_gpu --epochs 6 --fine-tune-mode last-linear-layer  --cos_sim_loss y --neg_rankings_loss y --task para --para_layers 2 --para_hidden_size 40 > para_test_3.txt
echo "Testing sts"
python3 multitask_classifier.py --use_gpu --epochs 6 --fine-tune-mode last-linear-layer --cos_sim_loss y --neg_rankings_loss y --task sts --sts_layers 0 --sts_hidden_size 20 > sts_test_1.txt
python3 multitask_classifier.py --use_gpu --epochs 6 --fine-tune-mode last-linear-layer  --cos_sim_loss y --neg_rankings_loss y --task sts --sts_layers 1 --sts_hidden_size 20 > sts_test_2.txt
python3 multitask_classifier.py --use_gpu --epochs 6 --fine-tune-mode last-linear-layer  --cos_sim_loss y --neg_rankings_loss y --task sts --sts_layers 2 --sts_hidden_size 20 > sts_test_3.txt