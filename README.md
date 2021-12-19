# Character-level Transformer Playground

The data is composed of names from 18 countries which are tokenized by individual letters (case sensitive, no [SOS] token), plus additional tokens such as [PAD], [CLS], [SEP] and [MASK].

This is no exact re-implementation of any kind of reference model but the training objectives are inspired from BERT and GPT. Model and training hyper-parameters can be edited in the ./configs/a_config.json as well as in the argument parser.

The "CLS" objective is a supervised classification which can be complemented with other self-supervised objectives.

The "MLM" objective is akin to BERT where some tokens are randomly masked or swapped and the bidirectional encoder must recover these.

The "LA" objective is akin to GPT where the upper diagonal look-ahead masking forces a left-to-right next token prediction based on attention over previous tokens only.

The "NSP" objective is akin to BERT where sequences are randomly split and the second half can be swapped with another sequence ending.


### Remarks

The combination of all losses at the same time may not be efficient. E.g. optimized BERT models such as RoBERTa have discarded the NSP loss. This approach is different than the usual SSL pre-training and supervised fine-tuning, however many fine-tuning approaches do keep SSL losses when possible. Setting detach_CLS=True corresponds to training the transformer encoder only on SSL losses and optimizing the supervised CLS loss with the prediction module alone.

If num_workers is low, the code may be quite slow given the inefficient batching methods implemented in data_names_utils.custom_collate_fn. Fast parallelized batch sampling can be achieved for objectives "CLS", "MLM" (if simplified) and "LA".


### Example run

Losses and accuracies while training a model on the 4 objectives together, with detach_CLS=False (e.g. config=train_4losses_config1.json).

![Alt text](training_runs/test1_4losses/separate_objectives.jpg'?raw=true "Logs of Separate Objectives")

![Alt text](training_runs/test1_4losses/total_loss.jpg'?raw=true "Logs of Total Losses")
