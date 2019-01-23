
time python main.py --data ./data/er.npz --save ./save/er.pk --log ./logs/er.log --exps 5 --patience 15 \
    --normalize 1 --loss mae --hidCNN 100 --hidRNN 100 --hidSkip 50 --output_fun no \
    --multi 1 --horizon 3 --highway_window 7 --window 14 --skip 7 --ps 3 

