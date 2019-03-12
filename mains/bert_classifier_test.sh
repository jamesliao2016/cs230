# Based on bert readme

export BERT_HOME_DIR=../bert
export BERT_BASE_DIR=../bert_data/cased_L-12_H-768_A-12
export GLUE_DIR=../glue_data
name="$1"
export OUTPUT_DIR=../experiments/bert/"$name"/

if (( $# <= 0 )); then
  echo 'Error: Provide task name arg'
  exit 1
fi

python $BERT_HOME_DIR/run_classifier.py \
  --task_name="$name" \
  --do_predict=true \
  --data_dir=$GLUE_DIR/"$name" \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir="$OUTPUT_DIR"
