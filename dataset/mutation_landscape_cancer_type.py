import pandas as pd
import numpy as np


class MutationLandscapeCancerType:
    """
    Loads data and processes from Sinkala et al. (2023)
    """

    processed_file = "mutations_across_cancer_types.xlsx"

    def __init__(
        self,
        mutations_across_cancer_file: str = "41598_2023_39608_MOESM3_ESM.xlsx",
        **kwargs
    ) -> None:
        """
        Initialize this class.

        :param mutations_across_cancer_file: supplementary data file.
        """
        self.__dict__.update(kwargs)

        self._mutations_across_cancer_file = mutations_across_cancer_file

    def load(self) -> pd.DataFrame:
        """
        load the data.
        """
        df = (
            pd.read_excel(
                "41598_2023_39608_MOESM3_ESM.xlsx",
                sheet_name="Each Gene Pair in Cancer Type",
            )
            .groupby("tumortype")
            .head(2)
            .sort_values(["tumortype", "pval"])
        )

        df["canonicalName"] = ""
        df["otherNames"] = ""
        df["definition"] = ""
        df["doids"] = ""

        df.loc[
            df["tumortype"] == "UCEC", "canonicalName"
        ] = "uterine corpus endometrial carcinoma"
        df.loc[df["tumortype"] == "UCEC", "otherNames"] = ""
        df.loc[
            df["tumortype"] == "UCEC", "definition"
        ] = "A uterine corpus cancer that is derived from the inner lining of the uterus."
        df.loc[df["tumortype"] == "UCEC", "doids"] = "0050939"

        df.loc[df["tumortype"] == "SKCM", "canonicalName"] = "skin cutaneous melanoma"
        df.loc[
            df["tumortype"] == "SKCM", "otherNames"
        ] = "malignant neck melanoma, cutaneous melanoma, malignant upper limb melanoma, malignant melanoma of skin of upper limb, malignant trunk melanoma, malignant melanoma of skin of lower limb, malignant melanoma of ear and/or external auricular canal, malignant scalp melanoma, malignant ear melanoma, malignant lower limb melanoma, malignant lip melanoma, malignant melanoma of skin of trunk except scrotum"
        df.loc[
            df["tumortype"] == "SKCM", "definition"
        ] = "A skin cancer that has material basis in melanocytes."
        df.loc[df["tumortype"] == "SKCM", "doids"] = "8923"

        # BLCA not in the serve actionable genes, although direct parent is.
        df.loc[df["tumortype"] == "BLCA", "canonicalName"] = "bladder carcinoma"
        df.loc[df["tumortype"] == "BLCA", "otherNames"] = "carcinoma of urinary bladder"
        df.loc[
            df["tumortype"] == "BLCA", "definition"
        ] = "A urinary bladder cancer that has material basis in abnormally proliferating cells derives from epithelial cells."
        df.loc[df["tumortype"] == "BLCA", "doids"] = "4007"

        # UCS not in the serve actionable genes.

        df.loc[
            df["tumortype"] == "OV", "canonicalName"
        ] = "ovarian serous cystadenocarcinoma"
        df.loc[df["tumortype"] == "OV", "otherNames"] = "serous cystadenoma"
        df.loc[
            df["tumortype"] == "OV", "definition"
        ] = "An ovary serous adenocarcinoma that has material basis in glandular epithelium, in which cystic accumulations of retained secretions are formed."
        df.loc[df["tumortype"] == "OV", "doids"] = "5746"

        df.loc[
            df["tumortype"] == "LUSC", "canonicalName"
        ] = "lung squamous cell carcinoma"
        df.loc[
            df["tumortype"] == "LUSC", "otherNames"
        ] = "epidermoid cell carcinoma of the lung, squamous cell carcinoma of lung"
        df.loc[
            df["tumortype"] == "LUSC", "definition"
        ] = "A non-small cell lung carcinoma that has material basis in the squamous cell."
        df.loc[df["tumortype"] == "LUSC", "doids"] = "3907"

        df.loc[df["tumortype"] == "STAD", "canonicalName"] = "gastric adenocarcinoma"
        df.loc[
            df["tumortype"] == "STAD", "otherNames"
        ] = "stomach adenocarcinoma, adenocarcinoma of stomach"
        df.loc[
            df["tumortype"] == "STAD", "definition"
        ] = "A stomach carcinoma that derives from epithelial cells of glandular origin."
        df.loc[df["tumortype"] == "STAD", "doids"] = "3717"

        df.loc[df["tumortype"] == "LUAD", "canonicalName"] = "lung adenocarcinoma"
        df.loc[
            df["tumortype"] == "LUAD", "otherNames"
        ] = "nonsmall cell adenocarcinoma, bronchogenic lung adenocarcinoma, adenocarcinoma of lung"
        df.loc[
            df["tumortype"] == "LUAD", "definition"
        ] = "A lung non-small cell carcinoma that derives from epithelial cells of glandular origin."
        df.loc[df["tumortype"] == "LUAD", "doids"] = "3910"

        df.loc[df["tumortype"] == "ESCA", "canonicalName"] = "esophagus adenocarcinoma"
        df.loc[df["tumortype"] == "ESCA", "otherNames"] = "oesophageal adenocarcinoma"
        df.loc[
            df["tumortype"] == "ESCA", "definition"
        ] = "An esophageal carcinoma that derives from epithelial cells of glandular origin."
        df.loc[df["tumortype"] == "ESCA", "doids"] = "4914"

        # DLBC not in the serve actionable genes, although direct parent is.
        df.loc[df["tumortype"] == "DLBCL", "canonicalName"] = "B-cell lymphoma"
        df.loc[df["tumortype"] == "DLBCL", "otherNames"] = "B-cell lymphocytic neoplasm"
        df.loc[
            df["tumortype"] == "DLBCL", "definition"
        ] = "A non-Hodgkin lymphoma that has material basis in B cells."
        df.loc[df["tumortype"] == "DLBCL", "doids"] = "707"

        df.loc[
            df["tumortype"] == "CESC", "canonicalName"
        ] = "cervical squamous cell carcinoma"
        df.loc[
            df["tumortype"] == "CESC", "otherNames"
        ] = "squamous cell carcinoma of the cervix uteri, squamous cell carcinoma of cervix"
        df.loc[
            df["tumortype"] == "CESC", "definition"
        ] = "A cervix carcinoma that has material basis in squamous cells of the cervix."
        df.loc[df["tumortype"] == "CESC", "doids"] = "3744"

        df.loc[
            df["tumortype"] == "HNSC", "canonicalName"
        ] = "head and neck squamous cell carcinoma"
        df.loc[
            df["tumortype"] == "HNSC", "otherNames"
        ] = "squamous cell carcinomas of head and neck, carcinoma of the head and neck, squamous cell carcinoma of the head and neck"
        df.loc[
            df["tumortype"] == "HNSC", "definition"
        ] = "A head and neck carcinoma that has material basis in squamous cells that line the moist, mucosal surfaces inside the head and neck."
        df.loc[df["tumortype"] == "HNSC", "doids"] = "5520"

        df.loc[df["tumortype"] == "SARC", "canonicalName"] = "sarcoma"
        df.loc[
            df["tumortype"] == "SARC", "otherNames"
        ] = "connective and soft tissue neoplasm, tumor of soft tissue and skeleton"
        df.loc[
            df["tumortype"] == "SARC", "definition"
        ] = "A cell type cancer that has material basis in abnormally proliferating cells derives from embryonic mesoderm."
        df.loc[df["tumortype"] == "SARC", "doids"] = "1115"

        df.loc[df["tumortype"] == "LIHC", "canonicalName"] = "hepatocellular carcinoma"
        df.loc[df["tumortype"] == "LIHC", "otherNames"] = "hepatoma"
        df.loc[
            df["tumortype"] == "LIHC", "definition"
        ] = "A liver carcinoma that has material basis in undifferentiated hepatocytes and located in the liver."
        df.loc[df["tumortype"] == "LIHC", "doids"] = "684"

        # BRCA not in the serve actionable genes, although direct parent is.
        df.loc[df["tumortype"] == "BRCA", "canonicalName"] = "breast adenocarcinoma"
        df.loc[
            df["tumortype"] == "BRCA", "otherNames"
        ] = "mammary adenocarcinoma, adenocarcinoma of breast"
        df.loc[
            df["tumortype"] == "BRCA", "definition"
        ] = "A breast carcinoma that originates in the milk ducts and/or lobules (glandular tissue) of the breast."
        df.loc[df["tumortype"] == "BRCA", "doids"] = "3458"

        df.loc[
            df["tumortype"] == "COADREAD", "canonicalName"
        ] = "colorectal adenocarcinoma"
        df.loc[df["tumortype"] == "COADREAD", "otherNames"] = ""
        df.loc[
            df["tumortype"] == "COADREAD", "definition"
        ] = "A colorectal carcinoma that derives from epithelial cells of glandular origin."
        df.loc[df["tumortype"] == "COADREAD", "doids"] = "0050861"

        df.loc[df["tumortype"] == "CHOL", "canonicalName"] = "cholangiocarcinoma"
        df.loc[
            df["tumortype"] == "CHOL", "otherNames"
        ] = "adult primary cholangiocellular carcinoma, cholangiosarcoma, adult primary cholangiocarcinoma"
        df.loc[
            df["tumortype"] == "CHOL", "definition"
        ] = "A bile duct adenocarcinoma that has material basis in bile duct epithelial cells."
        df.loc[df["tumortype"] == "CHOL", "doids"] = "4947"

        # ACC not in the serve actionable genes, although direct parent is.
        df.loc[df["tumortype"] == "ACC", "canonicalName"] = "cancer"
        df.loc[
            df["tumortype"] == "ACC", "otherNames"
        ] = "malignant neoplasm, primary cancer, malignant tumor"
        df.loc[
            df["tumortype"] == "ACC", "definition"
        ] = "A disease of cellular proliferation that is malignant and primary, characterized by uncontrolled cellular proliferation, local cell invasion and metastasis."
        df.loc[df["tumortype"] == "ACC", "doids"] = "162"

        df.loc[df["tumortype"] == "PAAD", "canonicalName"] = "pancreatic adenocarcinoma"
        df.loc[
            df["tumortype"] == "PAAD", "otherNames"
        ] = "pancreas adenocarcinoma, adenocarcinoma of the pancreas"
        df.loc[
            df["tumortype"] == "PAAD", "definition"
        ] = "A pancreatic carcinoma that derives from epithelial cells of glandular origin."
        df.loc[df["tumortype"] == "PAAD", "doids"] = "4074"

        df.loc[df["tumortype"] == "PRAD", "canonicalName"] = "prostate adenocarcinoma"
        df.loc[df["tumortype"] == "PRAD", "otherNames"] = ""
        df.loc[
            df["tumortype"] == "PRAD", "definition"
        ] = "A prostate carcinoma that derives from epithelial cells of glandular origin."
        df.loc[df["tumortype"] == "PRAD", "doids"] = "2526"

        df.loc[df["tumortype"] == "GBM", "canonicalName"] = "glioblastoma"
        df.loc[
            df["tumortype"] == "GBM", "otherNames"
        ] = "GBM, primary glioblastoma multiforme, adult glioblastoma multiforme, grade IV adult astrocytic tumor, spongioblastoma multiforme"
        df.loc[
            df["tumortype"] == "GBM", "definition"
        ] = "A malignant astrocytoma characterized by the presence of small areas of necrotizing tissue that is surrounded by anaplastic cells as well as the presence of hyperplastic blood vessels, and that has material basis in abnormally proliferating cells derives from multiple cell types including astrocytes and oligondroctyes."
        df.loc[df["tumortype"] == "GBM", "doids"] = "3068"

        df.loc[
            df["tumortype"] == "KIRP", "canonicalName"
        ] = "papillary renal cell carcinoma"
        df.loc[
            df["tumortype"] == "KIRP", "otherNames"
        ] = "chromophil carcinoma of kidney, papillary kidney carcinoma, sporadic papillary renal cell carcinoma"
        df.loc[
            df["tumortype"] == "KIRP", "definition"
        ] = "A renal cell carcinoma that is characterized by the development of multiple, bilateral papillary renal tumors."
        df.loc[df["tumortype"] == "KIRP", "doids"] = "4465"

        # KIRC not in the serve actionable genes, although direct parent is.
        df.loc[df["tumortype"] == "KIRC", "canonicalName"] = "renal cell carcinoma"
        df.loc[
            df["tumortype"] == "KIRC", "otherNames"
        ] = "adenocarcinoma of kidney, hypernephroma, RCC"
        df.loc[
            df["tumortype"] == "KIRC", "definition"
        ] = "A renal carcinoma that has material basis in the lining of the proximal convoluted renal tubule of the kidney."
        df.loc[df["tumortype"] == "KIRC", "doids"] = "4450"

        # MESO not applicable to co-occurring cancers
        # LGG not in serve actionable genes.

        # UVM not applicable to co-occurring cancers
        # PCPG not in the serve actionable genes

        # TGCT not in the serve actionable genes, although direct parent is.
        df.loc[df["tumortype"] == "TGCT", "canonicalName"] = "germ cell cancer"
        df.loc[
            df["tumortype"] == "TGCT", "otherNames"
        ] = "malignant tumor of the germ cell, germ cell tumour, germ cell neoplasm"
        df.loc[
            df["tumortype"] == "TGCT", "definition"
        ] = "A cell type cancer that has material basis in abnormally proliferating cells derives_from germ cells."
        df.loc[df["tumortype"] == "TGCT", "doids"] = "2994"

        # KICH not applicable to co-occurring cancers

        # THYM not in the serve actionable genes, although direct parent is.
        df.loc[df["tumortype"] == "THYM", "canonicalName"] = "thymus cancer"
        df.loc[
            df["tumortype"] == "THYM", "otherNames"
        ] = "thymic tumor, neoplasm of thymus, thymic neoplasm"
        df.loc[
            df["tumortype"] == "THYM", "definition"
        ] = "An immune system cancer located_in the thymus."
        df.loc[df["tumortype"] == "THYM", "doids"] = "3277"

        df.loc[df["tumortype"] == "LAML", "canonicalName"] = "acute myeloid leukemia"
        df.loc[
            df["tumortype"] == "LAML", "otherNames"
        ] = "acute myeloblastic leukaemia, AML, acute myeloblastic leukemia, acute myelogenous leukemia, acute myelogenous leukaemia, leukemia, myelocytic, acute, acute myeloid leukaemia"
        df.loc[
            df["tumortype"] == "LAML", "definition"
        ] = "A myeloid leukemia that is characterized by the rapid growth of abnormal white blood cells that accumulate in the bone marrow and interfere with the production of normal blood cells."
        df.loc[df["tumortype"] == "LAML", "doids"] = "9119"

        df.loc[df["tumortype"] == "THCA", "canonicalName"] = "thyroid gland carcinoma"
        df.loc[
            df["tumortype"] == "THCA", "otherNames"
        ] = "head and neck cancer, thyroid"
        df.loc[
            df["tumortype"] == "THCA", "definition"
        ] = "a thyroid gland cancer that has_material_basis_in epithelial cells."
        df.loc[df["tumortype"] == "THCA", "doids"] = "3963"

        df.to_excel(self.processed_file)

        return df
