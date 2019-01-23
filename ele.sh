
time python main.py --data ./data/ele.npz --save ./save/ele.pk --log ./logs/ele.log --exps 3 --patience 10 \
    --normalize 1 --loss mae --hidCNN 200 --hidRNN 200 --hidSkip 100 --output_fun linear \
    --multi 1 --horizon 3 --highway_window 12 --window 48 --skip 24 --ps 4

