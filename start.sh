export CUDA_VISIBLE_DEVICES=0,1,2,3
python ./train.py \
--dest_graph_output=./inceptionv3.pkl \
--colocate_grads_with_ops=True \
--model_name=inception_v3 \
--batch_size=256 \
--cost_path=./tmp/cost.pkl \
--costgen=False \
--only_forward=True \
--placement_method=m_sct \
--placer_type=aware \
--grappler_time=360000 \
--optimizer=adam \
--exe_pattern=aware_forward
