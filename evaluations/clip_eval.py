from evaluations.base import BaseEvaluator
import argparse
from CLIPScore_eval.CLIP_similarity import main


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="CLIP similarity evaluation.")
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        required=True,
        help="Path to read samples and output scores",
    )
    parser.add_argument(
        "--complex",
        type=bool,
        default=False,
        help="To evaluate on samples in complex category or not",
    )
    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args([])
    return args


class CLIPEvaluator(BaseEvaluator):
    def __init__(self, args):
        super(CLIPEvaluator, self).__init__()
        self.args = args

    def evaluate(self, out_dir):
        args = parse_args([f"--outpath={out_dir}"])
        score = main(args)
        return score