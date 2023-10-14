import pandas as pd


class AlternativeTreatmentNames:
    """
    Loads alternative names
    """

    def __init__(self) -> None:
        """
        Initialize this class.
        """
        self._df = None
        self._total_alternative_names = 0

    @property
    def df(self) -> pd.DataFrame:
        """
        Get the dataframe for this class.
        """
        if self._df is None:
            self.load()

        return self._df

    @property
    def stats(self) -> pd.DataFrame:
        """
        Get the stats
        """
        return pd.DataFrame(
            {"total_alternative_names": [self._total_alternative_names]}
        )

    def find_name(self, name: str) -> str | None:
        """
        Find the canonical name.
        """
        canonical_name = self._df.loc[
            self._df["alternative_name"] == name, "canonical_name"
        ]

        if len(canonical_name) == 0:
            return None

        self._total_alternative_names += 1
        return canonical_name.iloc[0]

    def load(self) -> pd.DataFrame:
        """
        load the data.
        """
        names = [
            ("gefitinib", ["iressa"]),
            ("ag-120", ["ivosidenib", "tibsovo"]),
            ("cetuximab", ["erbitux"]),
            ("aspirin", ["acetylsalicylic acid"]),
            ("erlotinib", ["tarceva"]),
            ("cobimetinib", ["cotellic"]),
            ("erdafitinib", ["balversa"]),
            ("ceritinib", ["zykadia"]),
            ("pemetrexed", ["alimta", "pemfexy", "ciambra", "pemrydi rtu"]),
            ("trametinib", ["mekinist"]),
            ("imatinib", ["gleevec", "glivec"]),
            ("daunorubicin", ["cerubidine", "vyxeos", "rubidomycin"]),
            ("afatinib", ["gilotrif", "giotrif"]),
            ("carboplatin", ["paraplatin"]),
            ("dabrafenib", ["tafinlar"]),
            ("enasidenib", ["idhifa"]),
            ("irinotecan", ["camptosar", "onivyde"]),
            ("decitabine", ["dacogen"]),
            ("ribociclib", ["kisqali"]),
            ("trastuzumab", ["herceptin"]),
            ("selumetinib", ["koselugo"]),
            ("abemaciclib", ["verzenio"]),
            ("olaparib", ["lynparza"]),
            ("panitumumab", ["vectibix"]),
            ("everolimus", ["afinitor", "zortress", "votubia"]),
            ("bevacizumab", ["avastin", "mvasi"]),
            ("fulvestrant", ["faslodex"]),
            ("paclitaxel", ["taxol", "abraxane"]),
            ("brigatinib", ["alunbrig"]),
            ("nilotinib", ["tasigna"]),
            ("nivolumab", ["opdivo", "opdualag"]),
            ("binimetinib", ["mektovi"]),
            ("alectinib", ["alecensa", "alecensaro"]),
            ("docetaxel", ["taxotere"]),
            ("midostaurin", ["rydapt"]),
            ("bgj398", ["infigratinib", "truseltiq"]),
            ("atezolizumab", ["tecentriq"]),
            ("vemurafenib", ["zelboraf"]),
            ("crizotinib", ["xalkori"]),
        ]

        self._df = pd.DataFrame(
            {
                "canonical_name": [x[0] for x in names],
                "alternative_name": [x[1] for x in names],
            }
        ).explode(column="alternative_name")

        return self._df
