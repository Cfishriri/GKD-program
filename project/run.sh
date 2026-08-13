cd /root/blockdata/project
CUDA_VISIBLE_DEVICES=0,1 python teacher-confidence-checking.py \
--student_model /root/eb-public/huggingface-models/Qwen/Qwen3-1.7B \
--teacher_model /root/eb-public/huggingface-models/Qwen/Qwen3-4B \
  --question "一个正整数 \(n\) 满足以下条件：  
\(n\) 除以 7 余 3；  
\(n\) 除以 11 余 5；  
\(n\) 除以 13 余 7；  
\(1000<n<5000\)。
求满足条件的最小 \(n\)。" \
  --correct_answer "1302" \
  --max_new_tokens 2048 \
  --output_dir ./recoverability_output