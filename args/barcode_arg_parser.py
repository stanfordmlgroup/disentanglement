from .train_arg_parser import TrainArgParser


class BarcodeArgParser(TrainArgParser):
    """Argument parser for args used only in weak disentangle/dcgan training."""
    def __init__(self):
        super(BarcodeArgParser, self).__init__()
        
        self.parser.add_argument('--real', default=False, action='store_true', help='Real, as opposed to fake dataset')
        self.parser.add_argument('--override', default=False, action='store_true', help='Override writing to existing json file')
        self.parser.add_argument('--bicluster', default=False, action='store_true', help='Bicluster for covariance or not')
        self.parser.add_argument('--search_n_clusters', default=False, action='store_true', help='Search for biclustering - how many clusters is appropriate for highest score')
        self.parser.add_argument('--verbose', default=False, action='store_true', help='Print extra things')
        self.parser.add_argument('--sup', default=False, action='store_true', help='Run supervised variant too.')
        self.parser.add_argument('--ones_only', default=False, action='store_true', help='Whether to include only ones when aggregating (e.g. celeba reals)')
        self.parser.add_argument('--save_interpolations', default=False, action='store_true', help='Save corresponding interpolations from fake dims that were matched to reals from mean biclustering, --bicluster needs to be on')
        self.parser.add_argument('--suffix', default=None, type=str, help='Add suffix to name file')
        self.parser.add_argument('--scores_file', default=None, type=str, help='Scores file')
        self.parser.add_argument('--L_0', default=None, type=int, help='Experimental landmark value for barcode gen')
        self.parser.add_argument('--gamma', default=None, type=float, help='Experimental gamma value for barcode gen')
        self.parser.add_argument('--save_scores', default=False, action='store_true', help='Whether to save scores to file')
        self.parser.add_argument('--gs_results_dir', default='/deep/group/disentangle/gs_results', type=str, help='Results dir to dump json files')
        self.parser.add_argument('--extra', default='real_dsprites', type=str, help='Barcode name for vis cov')
        self.parser.add_argument('--preprocess', default=False, action='store_true', help='Preprocess barcodes in vis_cov.')
        self.parser.add_argument('--plot', default=False, action='store_true', help='Plot visualizations.')
        self.parser.add_argument('--skip_all', default=False, action='store_true', help='Skip saving to all scores csv.')
