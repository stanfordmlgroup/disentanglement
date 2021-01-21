from .base_arg_parser import BaseArgParser


class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()

        self.parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train.')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 value for Adam optimizer.')
        self.parser.add_argument('--max_ckpts', type=int, default=3, help='Max ckpts to save.')
