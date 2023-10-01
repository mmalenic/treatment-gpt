from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from Bio import Entrez
import metapub

import time
from datetime import datetime

import re


class PubmedDownloader:
    """
    Downloads data from pubmed.
    """

    abstract_pdf_delay = 5
    pubmed_download_delay = 300
    pubmed_url_regex = re.compile(
        "(https|http)://(www\.)?(pubmed\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pubmed/)(?P<pubmed_id>\d+)/?"
    )

    def __init__(self, email: str, output_dir: str = "data/pubmed/", **kwargs) -> None:
        """
        Initialize this class.

        :param output_dir: output directory.
        """
        self.__dict__.update(kwargs)

        self._output_dir = output_dir
        self._datetime = None

        Entrez.email = email

    def download(
        self,
        pubmed_id: str,
        save_abstract_to: str | Path,
        save_pdf_to: Optional[str | Path] = None,
    ) -> str:
        """
        Download a pubmed id and save the abstracts and pdfs.
        """
        Path(self._output_dir).mkdir(exist_ok=True, parents=True)

        if Path(save_abstract_to).exists():
            return Path(save_abstract_to).read_text(encoding="utf-8")

        url_match = self.pubmed_url_regex.match(pubmed_id)
        if url_match:
            pubmed_id = url_match.group("pubmed_id")

        if pubmed_id is None or pubmed_id == "":
            raise Exception("pubmed id regex match failed")

        while (
            self._datetime is not None
            and (datetime.now() - self._datetime).total_seconds()
            < self.pubmed_download_delay
        ):
            print("waiting for delay to get next pubmed id:", pubmed_id)
            time.sleep(self.pubmed_download_delay)

        print("downloading pubmed id:", pubmed_id)

        handle = Entrez.efetch(db="pubmed", id=pubmed_id, retmode="xml")

        try:
            records = Entrez.read(handle)
            abstract = records["PubmedArticle"][0]["MedlineCitation"]["Article"][
                "Abstract"
            ]["AbstractText"][0]

            Path(save_abstract_to).write_text(abstract, encoding="utf-8")
        except Exception as e:
            print("failed to read pubmed id:", pubmed_id, "with error:", e)
            handle.close()

        time.sleep(self.abstract_pdf_delay)

        if save_pdf_to is not None:
            print("downloading pubmed pdf:", pubmed_id)
            url = metapub.FindIt(pubmed_id).url

            urlretrieve(url, save_pdf_to)

        self._datetime = datetime.now()

        return Path(save_abstract_to).read_text(encoding="utf-8")
