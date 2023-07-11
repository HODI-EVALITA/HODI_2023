# Homotransphobia Detection in Italian (HODI) @ EVALITA 2023 

Official repository of [HODI](https://hodi-evalita.github.io/), the **shared task on Homotransphobia Detection in Italian at [Evalita 2023](http://www.evalita.it/)**.

Fill out this [form](https://forms.gle/RYQ6m2hHvQZDx7vbA) to access the training and testing set, an email notification will be sent with instructions on how to download the data. 

---

**NOTE!** For participants using Windows as OS we suggest unzipping the files with WinRAR zip.

See the task guidelines in this repository and the [task web page](https://hodi-evalita.github.io/) for more details.


## Evaluation Script Installation

### Dependencies

To install the required python packages, run:

```
pip install -r requirements.txt
```

In the case of problems, try to run ```pip install --upgrade pip setuptools``` first.

## Evaluation Script Usage

The evaluation script can be used for evaluating the results both of Subtask A and B given a zip file containing submission files as input. For submission files formats check the [HODI Guidelines](https://github.com/HODI-EVALITA/HODI_2023_data/blob/main/HODI_2023_Task_Guidelines.pdf).

### Subtask A 

For running the evaluation script for Subtask A, you can run something like the following:

```bash
python compute_metrics.py \
--submission_path teamName.zip \
--gold_path HODI_2023_train_subtaskA.tsv \
--task a \
--output_path result.tsv 
```

### Subtask B

For running the evaluation script for Subtask B, you can run something like the following:

```bash
python compute_metrics.py \
--submission_path teamName.zip \
--gold_path HODI_2023_train_subtaskB.tsv \
--task b \
--output_path result.tsv 
```

## Baseline Script Usage

The baselines are a TF-IDF Logistic Regression for Subtask A and a random baseline for Subtask B.

For running the baseline script, you can run something like the following:

```bash
python compute_baseline.py \
--train_path train.tsv \
--test_path test.tsv \
--task a \
--output_path result.tsv 
```

## Reference

If you use the data or code please cite the following paper:

       @inproceedings{hodi2023overview,
        title = {{HODI at EVALITA 2023: Overview of the Homotransphobia
        Detection in Italian Task}},
        author = {Nozza, Debora and Cignarella,
        Alessandra Teresa and Damo, Greta  and Caselli, Tommaso and Patti, Viviana},
        booktitle = {{Proceedings of the Eighth Evaluation Campaign of
        Natural Language Processing and Speech Tools for Italian. Final
        Workshop (EVALITA 2023)}},
        publisher = {CEUR.org},
        year = {2023},
        month = {September},
        address = {Parma, Italy} }

## Contacts

If you find issues on the evaluation script, please contact **Debora Nozza**: [Twitter](https://twitter.com/debora_nozza) | [Github](https://github.com/dnozza) | [Webpage](https://deboranozza.com)


[![licensebuttons by-nc-sa](https://licensebuttons.net/l/by-nc-sa/3.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0)
