import torch
import os

import argparse
from readers import ner_reader
from datasets.NERDataset import NERDataset
from torch.utils.data import DataLoader
import configparser

from utils import batch_to_device, bool_flag, create_logger, dump_json
import numpy as np
import random
from torch.nn import CrossEntropyLoss
from models.optimization import get_scheduler, get_optimizer
import uuid
from models.NERModel import NERModel, load_tokenizer
from eval.metrics import compute_seqacc
import json






def main(args):

    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # find out device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)


    # check if output dir exists. if so, assign a new one
    if os.path.isdir(args.outdir):
        # create new output dir
        outdir = os.path.join(args.outdir, str(uuid.uuid4()))
    else:
        outdir = args.outdir


    # make the output dir
    os.makedirs(outdir)
    if args.save_best:
        os.makedirs(os.path.join(outdir, 'best_model'))

    # create a logger
    logger = create_logger(__name__, to_disk=True, log_file='{}/{}'.format(outdir, args.logfile))
    logger.info('Created new output dir {}'.format(outdir))
    logger.info('Running experiments on {}'.format(device))
    # get config with all data locations
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)

    print(config.sections())

    # load train data
    tokenizer = load_tokenizer(args.bert_model)
    logger.info('Inputs will be lower cased: {}'.format(tokenizer.do_lower_case))
    train_data = NERDataset(ner_reader.load_data(config.get('Files', '{}_train'.format(args.ds)),ds=args.ds), tokenizer=tokenizer, max_seq_len=args.max_seq_len, split_seqs=args.split_seqs)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.bs, collate_fn=train_data.collate)
    logger.info('Loaded {} train examples from {}'.format(len(train_data), args.ds))
    logger.info('Found {} labels: {}'.format(len(train_data.label2id), train_data.label2id))

    id2label = train_data.id2label

    # load dev data
    dev_data = NERDataset(ner_reader.load_data(config.get('Files','{}_dev'.format(args.ds)),ds=args.ds), tokenizer=tokenizer,
                          max_seq_len=args.max_seq_len, label2id=train_data.label2id, split_seqs=args.split_seqs)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.bs_prediction, collate_fn=dev_data.collate)
    logger.info('Loaded {} dev examples from {}'.format(len(dev_data), json.dumps(args.ds, indent=4)))

    # load test data
    if args.predict:
        test_data = NERDataset(ner_reader.load_data(config.get('Files', '{}_test'.format(args.ds)),ds=args.ds),
                               tokenizer=tokenizer, max_seq_len=args.max_seq_len, label2id=train_data.label2id, split_seqs=args.split_seqs)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.bs_prediction, collate_fn=test_data.collate)
        logger.info('Loaded {} test examples from {}'.format(len(test_data), args.ds))


    # load model
    if args.load_from_disk:
        model = NERModel(model_name_or_path=os.path.join(args.model_path, args.model_name), num_labels=train_data.num_predictable_labels).model
    else:
        model = NERModel(model_name_or_path=args.bert_model, num_labels=train_data.num_predictable_labels).model
    model.config.id2label = train_data.id2label
    model.config.label2id = train_data.label2id
    model.to(device)
    headline = '############# Model Arch #############'
    logger.info('\n{}\n{}\n'.format(headline, model))
    logger.info("Total number of params: {}".format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))

    # get loss
    ce_loss = CrossEntropyLoss(ignore_index=train_data.padding_label)

    # get optimizer
    optimizer = get_optimizer(model, lr=args.lr, eps=args.eps, decay=args.decay)

    # get lr schedule
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = args.warmup_frac * total_steps
    logger.info('Scheduler: {} with {} warmup steps'.format(args.scheduler, warmup_steps))
    scheduler = get_scheduler(optimizer, scheduler=args.scheduler, warmup_steps=warmup_steps, t_total=total_steps)

    loss_values = []
    global_step = 0

    best_dev_score = 0
    epoch = -1

    # save the tokenizer
    logger.info('Saving tokenizer')
    tokenizer.save_pretrained(outdir)
    epochs_since_last_improvement = 0
    for epoch in range(args.epochs):
        logger.info('Starting epoch {}'.format(epoch))

        total_loss = 0

        for step, batch in enumerate(train_dataloader):

            model.train()

            # batch to device
            batch = batch_to_device(batch, device)

            # clear gradients
            model.zero_grad()

            # perform forward pass
            output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            # compute loss
            loss = ce_loss(output[0].view(output[0].shape[1]*output[0].shape[0], output[0].shape[2]), batch['labels'].view(-1))
            total_loss += loss.item()

            # perform backward pass
            loss.backward()

            # clip the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.

            # take a step and update the model
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1


            # evaluate on dev
            if step > 0 and step % args.evaluation_step == 0:
                model.eval()
                dev_score, dev_report, _ = evaluate_on_dev(model=model, data_loader=dev_dataloader, device=device, label_map=id2label)
                logger.info('Epoch {}, global step {}/{}\ttrain loss: {:.5f}\t dev score: {:.5f}'.format(epoch,
                                                                                                     global_step,
                                                                                                     total_steps, total_loss/step,
                                                                                                    dev_score))


        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        # evaluate on dev after epoch is finished
        model.eval()
        dev_score, dev_report, _ = evaluate_on_dev(model=model, data_loader=dev_dataloader, device=device,
                                                   label_map=id2label)
        logger.info('End of epoch {}, global step {}/{}\ttrain loss: {:.5f}\t dev score: {:.5f}\ndev report {}'.format(epoch,
                                                                                                 global_step,
                                                                                                 total_steps,
                                                                                                 total_loss / step,
                                                                                                 dev_score, dev_report))
        epochs_since_last_improvement += 1
        if dev_score >= best_dev_score:
            epochs_since_last_improvement = 0
            logger.info('New dev score {:.5f} > {:.5f}'.format(dev_score, best_dev_score))
            best_dev_score = dev_score
            if args.save_best and args.early_stopping > 0:
                #save model
                logger.info('Saving model after epoch {} as best model to {}'.format(epoch, os.path.join(outdir, 'best_model')))
                save_model(os.path.join(outdir, 'best_model/model_{}.pt'.format(epoch)), model)


                if args.predict:
                    logger.info('Predicting test data with best model at end of epoch {}'.format(epoch))
                    model.eval()
                    test_score, test_report, test_predictions = evaluate_on_dev(model=model,
                                                                                data_loader=test_dataloader,
                                                                                device=device,
                                                                                label_map=id2label)
                    # dump to file
                    dump_json(fname=os.path.join(outdir, 'best_model/test_preds_{}.json'.format(epoch)),
                              data={'f1': test_score, 'report': test_report, 'predictions': test_predictions})
        logger.info(
            'Epochs since last improvement: {}'.format(epochs_since_last_improvement))
        if args.early_stopping >=0 and epochs_since_last_improvement > args.early_stopping:
            logger.info('Stopping training early because no improvement was observed in {} epochs'.format(args.early_stopping))
            break



    if args.predict:
        logger.info('Predicting test data at end of epoch {}'.format(epoch))
        model.eval()
        test_score, test_report, test_predictions = evaluate_on_dev(model=model, data_loader=test_dataloader,
                                                                    device=device,
                                                                    label_map=id2label)
        # dump to file
        dump_json(fname=os.path.join(outdir, 'test_preds_{}.json'.format(epoch)),
                  data={'f1': test_score, 'report': test_report, 'predictions': test_predictions})
        logger.info(
            'End of epoch {}, global step {}/{}\t test score: {:.5f}\ntest report {}'.format(epoch,
                                                                                                               global_step,
                                                                                                               total_steps,

                                                                                                               test_score,
                                                                                                               test_report))


    if not args.save_best or args.early_stopping <= 0:
        # save model
        logger.info('Saving model after epoch {} to {}'.format(epoch, os.path.join(outdir)))
        save_model(os.path.join(outdir, 'model_{}.pt'.format(epoch)), model)




def evaluate_on_dev(model, data_loader, label_map, device):
    model.eval()
    cropped_preds = []
    cropped_golds = []

    for step, batch in enumerate(data_loader):
        # batch to device
        batch = batch_to_device(batch, device)

        # perform forward pass
        with torch.no_grad():
            output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            out_probs = output[0].data.cpu()
            out_probs = out_probs.numpy()

            mask = batch['attention_mask']
            golds = batch['labels'].data.cpu().numpy()

            preds = np.argmax(out_probs, axis=-1).reshape(mask.size()).tolist()

            # only take into account the predictions for non PAD tokens
            valid_length = mask.sum(1).tolist()

            final_predict = []
            final_golds = []
            for idx, p in enumerate(preds):

                final_predict.append(p[:int(valid_length[idx])])
                final_golds.append(golds[idx][:int(valid_length[idx])])

            cropped_preds.extend(final_predict)
            cropped_golds.extend(final_golds)


    score, report = compute_seqacc(cropped_preds, cropped_golds, label_map)
    return score, report, cropped_preds




def save_model(outpath, model):
    outpath = '/'.join(outpath.split('/')[:-1])
    model.save_pretrained(outpath)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=str, default='/home/mareike/PycharmProjects/multiskill/code/config.cfg')
    parser.add_argument("--save_best", type=bool_flag, default=True)
    parser.add_argument("--predict", type=bool_flag, default=True)
    parser.add_argument("--outdir", type=str, default='output_test')
    parser.add_argument("--logfile", type=str, default='ner-model.log')
    parser.add_argument("--bert_model", type=str, default='from_config',
                        choices=['pt-bert', 'pt-biobert', 'pt-biobert_lower', 'bert-base-multilingual-cased', 'clinical-bert',
                                 'romanian-bert', 'romanian-bert_upper', 'beto-bert', 'biobert', 'biobert_upper', 'flaubert',  'bert-base-cased',
                                 'from_config', 'adx', 'ptbiobert'])
    parser.add_argument("--model_path", type=str, default='/home/mareike/PycharmProjects/anydomainbert/code/anydomainbert/out')
    parser.add_argument("--model_name", type=str,
                        default='checkpoint-20')
    parser.add_argument("--load_from_disk", type=bool_flag,  default=False)
    parser.add_argument("--split_seqs", type=bool_flag, default=True, help='if set to true, sequences are split rather than truncated')
    parser.add_argument("--bs", type=int, default=10)
    parser.add_argument("--bs_prediction", type=int, default=8)
    parser.add_argument("--ds", type=str, default='smcalflow_cs_seqlabeling',
                        choices=['smcalflow_cs_seqlabeling'])
    parser.add_argument("--epochs", type=int, default=2, help='Number of max epochs')
    parser.add_argument("--early_stopping", type=int, default=10, help='If positive, this is the patience for early stopping. Training will be stopped if number of max epochs is reached, or if dev score does not incrase in x epochs')
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default='warmuplinear')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument('--evaluation_step', type=int, default=10,
                        help="Evaluate every n training steps")
    args = parser.parse_args()
    main(args)