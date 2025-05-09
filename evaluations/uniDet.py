from UniDet_eval.spatial_2D_eval import main as spatial_2d_main
from UniDet_eval.numeracy_eval import main as numeracy_main
from evaluations.base import BaseEvaluator
import argparse


#obj detection
def parse_args(arglist=None):
    # breakpoint()
    parser = argparse.ArgumentParser(description="UniDet evaluation.")
    parser.add_argument(
        "--outpath",
        type=str,
        default="../examples/",
        help="Path to output score",
    )
    parser.add_argument(
        "--complex",
        type=bool,
        default=False,
        help="Prompt is simple structure or in complex category",
    )
    args = parser.parse_args(arglist)
    return args

class UniDeTEvaluator(BaseEvaluator):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.dataset == 'spatial' or args.dataset == 'complex':
            self.main = spatial_2d_main
        elif args.dataset == 'numeracy':
            self.main = numeracy_main
        else:
            raise ValueError("Invalid dataset. Choose either 'spatial' or 'numeracy'.")
        
    def evaluate(self, out_dir):
        is_complex = (self.args.dataset == 'complex')
        eval_args = parse_args([f"--outpath={out_dir}"])
        eval_args.complex = is_complex
        score = self.main(eval_args)
        return score