start=`date +%s`

python3 format_dataset.py

for i in `seq 0 5`;
do
    (
        python3 hw4_vl4kz_mzw7af.py $i
        python3 predict.py $i > "kfolds_$i.txt"
    )&
done
wait
end=`date +%s`
python3 average.py
echo $((end-start))
