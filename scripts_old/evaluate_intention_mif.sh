for tau in 1; do
    for mutation_on in --mutation_on; do
        for pred_func in $1; do
            for num_top in 1 3; do
                echo tau = $tau
                echo mutation_on = $mutation_on
                echo pred_func = $pred_func
                echo num_top = $num_top
                python -u scripts/evaluate_intention_mif.py --tau $tau  $mutation_on --pred_func $pred_func \
                --num_top_intentions $num_top --bidirectional | tee -a logs/evaluate_intention_mif.txt
            done
        done
    done
done