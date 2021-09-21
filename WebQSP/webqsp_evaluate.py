import sys
import json
import os
from components.utils import dump_json

def main():
    if len(sys.argv) != 3:
        print("Usage: python webqsp_evaluate.py goldData predAnswers")
        sys.exit(-1)

    predAnswers = open(sys.argv[2]).readlines()
    
    legacy_format = []
    for item in predAnswers:
        item = json.loads(item)
        legacy_format.append({"QuestionId": item["qid"], "Answers": item["answer"]})

    tmp_filename = "misc/tmp_legacy_pred.json"
    dump_json(legacy_format, tmp_filename)
    os.system("python2 legacy_eval.py {} {}".format(sys.argv[1], tmp_filename))
    os.remove(tmp_filename)

if __name__ == "__main__":
    main()
