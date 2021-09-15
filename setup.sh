PWD=$(pwd)
echo ${PWD}

setup_dataset () {
    DATASET=$1
    echo "SETTING UP"${DATASET}
    ln -s ${PWD}/framework/components ${PWD}/${DATASET}
    ln -s ${PWD}/framework/models ${PWD}/${DATASET}
    ln -s ${PWD}/framework/executor ${PWD}/${DATASET}
    ln -s ${PWD}/framework/ontology ${PWD}/${DATASET}

    ln -s ${PWD}/framework/run_ranker.py ${PWD}/${DATASET}/
    ln -s ${PWD}/framework/run_generator.py ${PWD}/${DATASET}/
}

setup_dataset GrailQA
setup_dataset WebQSP
