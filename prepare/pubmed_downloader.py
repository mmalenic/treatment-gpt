import os.path
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
    pubmed_download_delay = 60
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
        save_abstract_to: Optional[str | Path] = None,
        save_pdf_to: Optional[str | Path] = None,
    ) -> str:
        """
        Download a pubmed id and save the abstracts and pdfs.
        """
        Path(self._output_dir).mkdir(exist_ok=True, parents=True)

        if save_abstract_to is None:
            save_abstract_to = os.path.join(self._output_dir, f"{pubmed_id}")
        if save_pdf_to is None:
            save_pdf_to = os.path.join(self._output_dir, f"{pubmed_id}.pdf")

        if Path(save_abstract_to).exists():
            print("loading pubmed id from file:", pubmed_id)
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
            ]["AbstractText"]

            output_abstract = ""
            for part in abstract:
                try:
                    label = part.attributes["Label"].lower().title()
                    if label != "Unlabelled":
                        output_abstract += label
                        output_abstract += ": "

                    output_abstract += part
                    output_abstract += "\n"
                except (AttributeError, KeyError) as _:
                    output_abstract += part
                    output_abstract += "\n"

            Path(save_abstract_to).write_text(output_abstract, encoding="utf-8")
        except Exception as e:
            print("failed to read pubmed id:", pubmed_id, "with error:", e)
            Path(save_abstract_to).write_text("", encoding="utf-8")
            handle.close()

        time.sleep(self.abstract_pdf_delay)

        if save_pdf_to is not None:
            try:
                url = metapub.FindIt(pubmed_id).url

                if url is not None:
                    print(f"downloading pubmed pdf: {pubmed_id}, url: {url}")
                    urlretrieve(url, save_pdf_to)
                else:
                    print("no pubmed article pdf found for:", pubmed_id)
            except Exception as e:
                print("failed to download pdf of id:", pubmed_id, "with error:", e)

        self._datetime = datetime.now()

        return Path(save_abstract_to).read_text(encoding="utf-8")
