import argparse
import os
import time

from utils import load_txt, write_json_file

from paddlenlp import Taskflow
from paddlenlp.utils.log import logger


def main(args):
    """
    Predict based on Taskflow.
    """
    start_time = time.time()
    # read file
    logger.info("Trying to load dataset: {}".format(args.file_path))
    if not os.path.exists(args.file_path):
        raise ValueError("something with wrong for your file_path, it may not exist.")
    examples = load_txt(args.file_path)

    # define Taskflow for sentiment analysis
    schema = eval(args.schema)
    if args.load_from_dir:
        senta = Taskflow(
            "sentiment_analysis",
            model=args.model,
            schema=schema,
            aspects=args.aspects,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            task_path=args.load_from_dir,
        )
    else:
        senta = Taskflow(
            "sentiment_analysis",
            model=args.model,
            schema=schema,
            aspects=args.aspects,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )

    # predict with Taskflow
    logger.info("Start to perform sentiment analysis for your dataset, this may take some time.")
    results = senta(examples)

    # save results
    save_path = args.save_path
    if not save_path:
        save_dir = os.path.dirname(args.file_path)
        save_path = os.path.join(save_dir, "sentiment_results.json")
    write_json_file(results, save_path)
    logger.info("The results of sentiment analysis has been saved to: {}".format(save_path))
    logger.info("This run take {} seconds.".format(time.time() - start_time))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, default="./data/test_hotel.txt", help="The file path that you want to perform sentiment analysis on.")
    parser.add_argument("--save_path", type=str, default="./data/sentiment_analysis.json", help="The saving path for the results of sentiment analysis.")
    parser.add_argument("--model", choices=['uie-senta-base', 'uie-senta-medium', 'uie-senta-mini', 'uie-senta-micro', 'uie-senta-nano'], default="uie-senta-base", help="The model name that you wanna use for sentiment analysis.")
    parser.add_argument("--load_from_dir", default=None, type=str, help="The directory path for the finetuned model to predict, if set None, it will download model according to model_name.")
    parser.add_argument("--schema", default="[{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]", type=str, help="The schema for UIE to extract infomation.")
    parser.add_argument("--aspects", default=None, type=str, nargs="+", help="A list of pre-given aspects, that is to say, Pipeline only perform sentiment analysis on these pre-given aspects if you input it.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization.")

    args = parser.parse_args()
    # yapf: enable
    # args.aspects = ['存储','续航','性能','信号','质感','物流','送货','音质','性价比','包装','客服','充电','屏幕','手感']
    main(args)
