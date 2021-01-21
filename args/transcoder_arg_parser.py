from .train_arg_parser import TrainArgParser


class TranscoderArgParser(TrainArgParser):
    """ Argument parser for args used only in transcoder training """
    def __init__(self):
        super().__init__()

        self.parser.add_argument('--cycle', action='store_true', default=False,
                                 help='Train with cycle-consistent loss')
        self.parser.add_argument('--cycle_method', default='simultaneous',
                                 choices=['simultaneous', 'alternating'],
                                 help='Cycle-consistent training method')
        self.parser.add_argument('--fwd_lambda', type=float, default=0.1,
                                 help='Cycle loss weighting for forward cycle')
        self.parser.add_argument('--rev_lambda', type=float, default=0.01,
                                 help='Cycle loss weighting for reverse cycle')

        self.parser.add_argument('-use_both_datasets', action='store_true', help='Whether to also use the real dataset in training the MLP.')
        self.parser.add_argument('-mega', action='store_true', help='Whether to train with the mega loop.')

