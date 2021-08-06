for lr in 1e-4 ; do
    for num_lstms in 1; do
        for bidirectional in --bidirectional; do
            for end_mask in ""; do
                for dataset_ver in 0 25 50 75; do
                    echo lr = $lr
                    echo num_lstms = $num_lstms
                    echo hidden_size = $hidden_size
                    echo dataset_ver = $dataset_ver
                    echo bidirectional = $bidirectional
                    CUDA_VISIBLE_DEVICES=1 python -u scripts/main_wlstm.py \
                        --dataset_ver $dataset_ver --mode train --num_epochs 200 \
                        --save_epochs 50 --compute_baseline --batch_size 64 \
                        --num_lstms $num_lstms --lr $lr \
                        $bidirectional $end_mask | tee -a logs/train_wlstm.txt
                done
            done
        done
    done
done
