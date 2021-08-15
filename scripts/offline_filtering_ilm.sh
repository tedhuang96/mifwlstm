num_lstms=1
bidirectional=--bidirectional
for tau in 1; do
    for prediction_method in ilm; do
        for mutable in --mutable; do
            echo tau = $tau
            echo prediction_method = $prediction_method
            echo mutable = $mutable
            python -u scripts/offline_filtering_process.py --tau $tau --prediction_method $prediction_method \
                $mutable | tee -a logs/offline_filtering.txt
        done
    done
done