
time python main.py --data ./data/ele.npz --save ./save/ele.pk --log ./logs/ele.log --exps 3 --patience 10 \
    --loss mae --hidCNN 100 --hidRNN 100 --hidSkip 50 --output_fun linear \
    --multi multi --horizon 3 --highway_window 12 --window 48 --skip 24 --ps 4 

