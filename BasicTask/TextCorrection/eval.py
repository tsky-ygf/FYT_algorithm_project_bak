# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import sys
from turtle import ScrolledCanvas

from regex import E


sys.path.append("..")
sys.path.append('.')
import pycorrector
from pycorrector.utils import eval

pwd_path = os.path.abspath(os.path.dirname(__file__))

import time
def eval_by_model(input_file_path='', verbose=True):
    """
    Args:
        input_file_path:
        verbose:

    Returns:
        Acc, Recall, F1
    """
    print(input_file_path)
    from pycorrector.macbert.macbert_corrector import MacBertCorrector
    model = MacBertCorrector()
    pycorrector.set_custom_confusion_dict('/home/fyt/chi/FYT_algorithm_project/data/text_corrector/custom_confusion.txt')
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    neg_idx_list = []
    pos_idx_list = []
    total_num = 0
    start_time = time.time()

    tgt_list = []
    src_list = []
    model_tgt_pred_list = []
    rule_tgt_pred_list = []

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            src = parts[0]
            tgt = parts[1]

            #召回
            tgt_pred, pred_detail = model.macbert_correct(src)#模型
            rule_tgt_pred, rule_pred_detail = pycorrector.correct(src)#规则

            #将纠错结果写入文件
            src_list.append(src)
            tgt_list.append(tgt)

            model_tgt_pred_list.append(tgt_pred)
            rule_tgt_pred_list.append(rule_tgt_pred)

            #排序/融合,先规则纠错，如果不存在纠错的情况，再采用模型纠错的结果

            if rule_tgt_pred != src:
                tgt_pred = rule_tgt_pred
                pred_detail = rule_pred_detail
            else:
                pass
            

            if verbose:
                print()
                print('input  :', src)
                print('truth  :', tgt)
                print('predict:', tgt_pred, pred_detail)

            # 负样本
            if src == tgt:
                # 预测也为负
                if tgt == tgt_pred:
                    TN += 1
                    print('right')
                # 预测为正
                else:
                    FP += 1
                    neg_idx_list.append(idx)
                    print('wrong')
            # 正样本
            else:
                # 预测也为正
                if tgt == tgt_pred:
                    TP += 1
                    pos_idx_list.append(idx)
                    print('right')
                # 预测为负
                else:
                    FN += 1
                    print('wrong')
            total_num += 1
        spend_time = time.time() - start_time
        acc = (TP + TN) / total_num
        precision = TP / (TP + FP) if TP > 0 else 0.0
        recall = TP / (TP + FN) if TP > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        print(
            f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
            f'cost time:{spend_time:.2f} s, total num: {total_num}')
        print(f'误纠个数:{len(neg_idx_list)}, 正确纠错个数:{len(pos_idx_list)}')

        import pandas as pd
        df = pd.DataFrame({'src': src_list, 'tgt':tgt_list, 'model_tgt_pred':model_tgt_pred_list, 'rule_tgt_pred':rule_tgt_pred_list})
        tgt_file_path = '.'.join(input_file_path.split('.')[:-1]) + '_pred.xlsx'
        df.to_excel(tgt_file_path, index=False)
        tgt_file_path = '.'.join(input_file_path.split('.')[:-1]) + '_pred.csv'
        df.to_csv(tgt_file_path, sep = '\t', index = False)
        return acc, precision, recall, f1



def main(args):
    if args.data == 'sighan_15' and args.model == 'rule':
        # Sentence Level: acc:0.5100, precision:0.5139, recall:0.1363, f1:0.2154, cost time:1464.87 s
        eval.eval_sighan2015_by_model(pycorrector.correct)
    if args.data == 'sighan_15' and args.model == 'bert':
        # right_rate:0.37623762376237624, right_count:38, total_count:101;
        # recall_rate:0.3645833333333333, recall_right_count:35, recall_total_count:96, spend_time:503 s
        from pycorrector.bert.bert_corrector import BertCorrector
        model = BertCorrector()
        eval.eval_sighan2015_by_model(model.bert_correct)
    if args.data == 'sighan_15' and args.model == 'macbert':
        from pycorrector.macbert.macbert_corrector import MacBertCorrector
        model = MacBertCorrector()
        input_file_path = './data/text_corrector/nblh/test.csv'

        eval.eval_sighan2015_by_model_batch(model.batch_macbert_correct, input_file_path)
        # Sentence Level: acc:0.7900, precision:0.8250, recall:0.7293, f1:0.7742, cost time:4.90 s
    if args.data == 'sighan_15' and args.model == 'ernie':
        # right_rate:0.297029702970297, right_count:30, total_count:101;
        # recall_rate:0.28125, recall_right_count:27, recall_total_count:96, spend_time:655 s
        from pycorrector.ernie.ernie_corrector import ErnieCorrector
        model = ErnieCorrector()
        eval.eval_sighan2015_by_model(model.ernie_correct)
    if args.data == 'sighan_15' and args.model == 't5':
        from pycorrector.t5.t5_corrector import T5Corrector
        model = T5Corrector()
        eval.eval_sighan2015_by_model_batch(model.batch_t5_correct)
        # Sentence Level: acc:0.7582, precision:0.8321, recall:0.6390, f1:0.7229, cost time:5.12 s
    if args.data == 'sighan_15' and args.model == 'copyt5':
        from pycorrector.t5.copyt5_corrector import CopyT5Corrector
        model = CopyT5Corrector()
        eval.eval_sighan2015_by_model_batch(model.batch_t5_correct)
        # Sentence Level: acc:0.7255, precision:0.7648, recall:0.6409, f1:0.6974, cost time:28.58 s, total num: 1100
    if args.data == 'sighan_15' and args.model == 'convseq2seq':
        from pycorrector.seq2seq.seq2seq_corrector import Seq2SeqCorrector
        model = Seq2SeqCorrector()
        eval.eval_sighan2015_by_model_batch(model.seq2seq_correct)
        # Sentence Level: acc:0.3545, precision:0.2415, recall:0.1436, f1:0.1801, cost time:404.95 s
    if args.data == 'sighan_15' and args.model == 'bartseq2seq':
        from transformers import BertTokenizerFast
        from textgen import BartSeq2SeqModel
        tokenizer = BertTokenizerFast.from_pretrained('shibing624/bart4csc-base-chinese')
        model = BartSeq2SeqModel(
            encoder_type='bart',
            encoder_decoder_type='bart',
            encoder_decoder_name='shibing624/bart4csc-base-chinese',
            tokenizer=tokenizer,
            args={"max_length": 128})
        eval.eval_sighan2015_by_model_batch(model.predict)
        # Sentence Level: acc:0.6845, precision:0.6984, recall:0.6354, f1:0.6654

    if args.data == 'corpus500' and args.model == 'rule':
        # right_rate:0.486, right_count:243, total_count:500;
        # recall_rate:0.18, recall_right_count:54, recall_total_count:300, spend_time:78 s
        eval.eval_corpus500_by_model(pycorrector.correct)
    if args.data == 'corpus500' and args.model == 'bert':
        # right_rate:0.586, right_count:293, total_count:500;
        # recall_rate:0.35, recall_right_count:105, recall_total_count:300, spend_time:1760 s
        from pycorrector.bert.bert_corrector import BertCorrector
        model = BertCorrector()
        eval.eval_corpus500_by_model(model.bert_correct)
    if args.data == 'corpus500' and args.model == 'macbert':
        # Sentence Level: acc:0.724000, precision:0.912821, recall:0.595318, f1:0.720648, cost time:6.43 s
        from pycorrector.macbert.macbert_corrector import MacBertCorrector
        model = MacBertCorrector()
        eval.eval_corpus500_by_model(model.macbert_correct)
    
    if args.data == 'corpus500' and args.model == 'ernie':
        # right_rate:0.598, right_count:299, total_count:500;
        # recall_rate:0.41333333333333333, recall_right_count:124, recall_total_count:300, spend_time:6960 s
        from pycorrector.ernie.ernie_corrector import ErnieCorrector
        model = ErnieCorrector()
        eval.eval_corpus500_by_model(model.ernie_correct)
    
    
    if args.data == 'nblh' and args.model == 'rule':#测试南北联合的语料
        input_file_path = './data/text_corrector/nblh/test.csv'
        eval_by_model(input_file_path)
        
    if args.data == 'nblh' and args.model == 'both':
        input_file_path = './data/text_corrector/nblh/test.csv'
        eval_by_model(input_file_path)

    if args.data == 'nblh' and args.model == 'nblh_macbert':#测试南北联合的语料
        input_file_path = './data/text_corrector/nblh/test.csv'
        eval_by_model(input_file_path)

    if args.data == 'sighan_15' and args.model == 'nblh_macbert':
        from BasicTask.TextCorrection.macbert.predict import MacbertCorrected
        model = MacbertCorrected()
        input_file_path = './data/text_corrector/sighan_2015/test.tsv'
        eval.eval_sighan2015_by_model(model.__call__, input_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='nblh', help='evaluate dataset, sighan_15/corpus500')
    parser.add_argument('--model', type=str, default='both', help='which model to evaluate, rule/bert/macbert/ernie')
    args = parser.parse_args()
    main(args)