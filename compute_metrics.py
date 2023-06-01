import argparse
import zipfile
import pandas as pd
import evaluation_utils
def read_input(submission_path, task):
    data_predicted = {}
    column_names = ''
    converters_raw = {}

    if submission_path.endswith(".zip"):
        unzipped = zipfile.ZipFile(submission_path, "r")
    else:
        raise Exception("Exception: submission file is not a ZIP file.")
    if not ((task.lower() == "a") or (task.lower() == "b")):
        raise Exception('Exception: task should be either "a" or "b".')
    elif task.lower() == "a":
        converters_raw = {'id': str, 'homotransphobic': int}
    elif task.lower() == "b":
        converters_raw = {'id': str, 'rationales': lambda x: eval(x)}

    run_filelist = [f for f in unzipped.namelist() if
                    ("run" in f) and ("." + task + "." in f.lower()) and (not ('__MACOSX' in f))]

    for runs_admitted in ['run1', 'run2', 'run3']:
        if len([f for f in run_filelist if (runs_admitted in f)]) == 0:
            continue
        data_path = [f for f in run_filelist if
                         (runs_admitted in f) and ("." + task + "." in f.lower())]
        if len(data_path) == 1:
            try:
                data_predicted[runs_admitted] = pd.read_csv(unzipped.open(data_path[0]), sep="\t", index_col=None, converters=converters_raw)
            except:
                raise Exception('Exception: error in raw prediction file format.')
        elif len(data_path) > 1:
            raise Exception('Exception: too many raw prediction files for ' + runs_admitted + ' (check file names).')
        elif len(data_path) == 0:
            raise Exception('Exception: no raw prediction files for ' + runs_admitted + ' (check file names).')

    return data_predicted


def read_gold(gold_path_raw, task):
    if task == 'a':
        data_gold = pd.read_csv(gold_path_raw, sep="\t",
                                    converters={'id': str, 'text': str, 'homotransphobic': int})
    elif task == 'b':
        data_gold = pd.read_csv(gold_path_raw, sep="\t", converters={'id': str, 'text': str, 'rationales': str})
    return data_gold

def evaluate(data_predicted, data_gold, output_path, task):
    print("Starting evaluation Subtask",task)
    f = open(output_path, "w")
    f.write("run_subtask"+task+"\tscore\n")
    for k in data_predicted.keys():
        print('*'*20,k)
        if task.lower() == "a":
            metric = evaluation_utils.evaluate_task_a_singlefile(data_predicted[k], data_gold, task)
        elif task.lower() == "b":
            metric = evaluation_utils.evaluate_task_b_singlefile(data_predicted[k], data_gold, task)
        else:
            raise Exception('Task should be either "a" or "b"')
        f.write(k+"\t"+str(metric)+"\n")
    f.close()
    print("Evaluation for Subtask "+task+" completed and saved at:",output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVALITA HODI 2023 - Evaluation script.')
    parser.add_argument('--submission_path', type=str, required=True,
                        help='path of the submission file (ZIP format)')
    parser.add_argument('--gold_path_raw', type=str, required=True,
                        help='path of gold raw tsv file')
    parser.add_argument('--task', type=str, required=True, choices=['a', 'b'],
                        help='task you want to evaluate ("a" or "b")')
    parser.add_argument('--output_path', type=str, required=False, default="result.tsv",
                        help='path of output result file')

    args = parser.parse_args()

    data_gold = read_gold(args.gold_path_raw, args.task)
    data_predicted = read_input(args.submission_path, args.task)

    evaluate(data_predicted, data_gold, args.output_path, args.task)
