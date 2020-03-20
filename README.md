# Dialogue Response Selection

### run experiment
```angular2
python run_classifier.py \\
       --task_name={advising, ubuntu} \\
       --model_name={bert-base-uncased, rnn} \\
       --output_dir=temp \\
       --data_dir={data_dir} \\
       --learning_rate=0.001 \\
       --num_train_epoch=50 \\
       --train_batch_size=128 \\
       --do_train \\
       --do_train_eval \\
       --tb_log_dir={tensorboard log dir}
```

