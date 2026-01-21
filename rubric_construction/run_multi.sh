#!/bin/bash

export JAVA_HOME=jdk-21.0.7
export CLASSPATH=.:${JAVA_HOME}/lib
export PATH=${CLASSPATH}:${JAVA_HOME}/bin:$PATH

retriever_url="http://localhost:8000"
embedder_model=""

seg2q_model=""
seg2q_model_api_url="http://localhost:30000/v1"
seg2q_model_api_key="EMPTY"

seg2q_judge_model=""
seg2q_judge_model_api_url="http://localhost:30000/v1"
seg2q_judge_model_api_key="EMPTY"

seg_judge_model=""
seg_judge_model_api_url="http://localhost:30000/v1"
seg_judge_model_api_key="EMPTY"

nuggetizer_model=""
nuggetizer_model_api_url="http://localhost:30000/v1"
nuggetizer_model_api_key="EMPTY"

k_retrieved_passages=10
hard_k_retrieved_passages=20
threshold_score=0.60
max_new_queries=3
max_queries_per_passage=3
max_nuggets_per_segment=3
max_depth=30
max_total_nodes=300

total_save_path=""

for i in {0..99}; do
    train_set_path="reward_bench.modified.${i}.jsonl"
    save_path="reward_bench.rubrics.${i}.jsonl"
    log_path="reward_bench.rubrics.${i}.log"

    echo "running worker ${i}"

    python -u loop.py \
        --train_set_path $train_set_path \
        --retriever_url $retriever_url \
        --embedder_model $embedder_model \
        --seg2q_model $seg2q_model \
        --seg2q_model_api_url $seg2q_model_api_url \
        --seg2q_model_api_key $seg2q_model_api_key \
        --seg2q_judge_model $seg2q_judge_model \
        --seg2q_judge_model_api_url $seg2q_judge_model_api_url \
        --seg2q_judge_model_api_key $seg2q_judge_model_api_key \
        --seg_judge_model $seg_judge_model \
        --seg_judge_model_api_url $seg_judge_model_api_url \
        --seg_judge_model_api_key $seg_judge_model_api_key \
        --nuggetizer_model $nuggetizer_model \
        --nuggetizer_model_api_url $nuggetizer_model_api_url \
        --nuggetizer_model_api_key $nuggetizer_model_api_key \
        --k_retrieved_passages $k_retrieved_passages \
        --hard_k_retrieved_passages $hard_k_retrieved_passages \
        --threshold_score $threshold_score \
        --max_new_queries $max_new_queries \
        --max_queries_per_passage $max_queries_per_passage \
        --max_nuggets_per_segment $max_nuggets_per_segment \
        --max_depth $max_depth \
        --max_total_nodes $max_total_nodes \
        --save_path $save_path \
        --total_save_path $total_save_path > $log_path 2>&1 &
    
    sleep 30
done

wait
echo "all workers finished"


