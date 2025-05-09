from BLIPvqa_eval.BLIP_vqa import VQA_main, Create_annotation_for_BLIP, main
import argparse
from evaluations.base import BaseEvaluator


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="BLIP vqa evaluation.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        required=True,
        help="Path to output BLIP vqa score",
    )
    parser.add_argument(
        "--np_num",
        type=int,
        default=8,
        help="Noun phrase number, can be greater or equal to the actual noun phrase number",
    )
    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args([])
    return args


class BLIPEvaluator(BaseEvaluator):
    def __init__(self,args):
        super(BLIPEvaluator, self).__init__()
        self.args = args
    
    def evaluate(self, out_dir):
        args = parse_args([f"--out_dir={out_dir}"])
        score = main(args)
        return score

