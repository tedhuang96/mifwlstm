for tau in 1; do
    for mutation_on in --mutation_on; do
        for pred_func in $1; do
            for percent in {1..10}; do
                echo tau = $tau
                echo mutation_on = $mutation_on
                echo pred_func = $pred_func
                echo percent = $percent
                python -u scripts/automate_filtering.py --tau $tau  $mutation_on --pred_func $pred_func \
                --filter_test_data $percent --bidirectional | tee -a logs/run_mif.txt
            done
        done
    done
done