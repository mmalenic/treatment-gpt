from classifier.baseline.all_protect import AllProtect
from dataset.mutation_landscape_cancer_type import MutationLandscapeCancerType
from prepare.all_samples import AllSamples


class RunBaseline:
    def __init__(self):
        """
        Initialize this class.
        """
        self._all_samples = AllSamples()
        self._cancer_types = MutationLandscapeCancerType()
        self._cancer_types.load()

        self._all_protect = AllProtect(doids=self._cancer_types.doids())

    def prepare_samples(self):
        """
        Fetch all samples.
        """
        self._all_samples.prepare()

    def run_protect(self):
        """
        Run protect for all samples.
        """
        self._all_protect.run()

    def run(self):
        """
        Run the baseline protect model.
        """
        self.prepare_samples()
        self.run_protect()
