import urllib.request
from pathlib import Path


class PrepareServe:
    characteristics = "ActionableCharacteristics.38.tsv"
    fusions = "ActionableFusions.38.tsv"
    genes = "ActionableGenes.38.tsv"
    hla = "ActionableHLA.38.tsv"
    hotspots = "ActionableHotspots.38.tsv"
    ranges = "ActionableRanges.38.tsv"

    def __init__(self, actionability_url: str, output_dir: str = "data/serve/"):
        self.actionability_url = actionability_url
        self.output_dir = output_dir

    def fetch(self):
        self.fetch_and_write(self.characteristics)
        self.fetch_and_write(self.fusions)
        self.fetch_and_write(self.genes)
        self.fetch_and_write(self.hla)
        self.fetch_and_write(self.hotspots)
        self.fetch_and_write(self.ranges)

    def fetch_and_write(self, name: str):
        output_file = Path(self.output_dir + name)

        if not output_file.exists():
            output_file.parent.mkdir(exist_ok=True, parents=True)

            output = urllib.request.urlopen(self.actionability_url + name).read()
            output_file.write_text(output)
