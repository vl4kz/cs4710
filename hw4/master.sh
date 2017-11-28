start=`date +%s`

python3 format_dataset.py

for i in `seq 0 5`;
do
    python3 hw4_vl4kz_mzw7af.py $i &
done
end=`date +%s`
echo $((end-start))

wait
#python3 predict.py > k_fold_error_rates.txt
