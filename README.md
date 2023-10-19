# Treatment GPT
GPT models for treatment recommendation based on PROTECT outputs.

To run PROTECT:
```sh
python -m run_baseline --mode "run"
```

To run the GPT models:
```sh
python -m run_classifer --api-key <api-key> --pubmed_email <email>
```

Or, use the jupyter notebook [`run.ipynb`][run].

[run]: run.ipynb