#inference for task with entity name
bash scripts/mlt_with_entity.sh your_cuda_devices your_data_name your_model_path your_output_folder tensor_parallel_size


#inference for task without entity name
bash scripts/mlt_without_entity.sh your_cuda_devices your_data_name your_model_path  your_output_folder tensor_parallel_size

#inference for biored
bash scripts/biored_generation.sh your_cuda_devices your_data_name your_model_path  your_output_folder tensor_parallel_size


#compute score for multi-class task
python eval/conpute_score_direct_prompt_multi_class.py --data_dir your_output_file_name

#compute score for binary-class task
python eval/conpute_score_direct_prompt_binary_class.py --data_dir your_output_file_name 