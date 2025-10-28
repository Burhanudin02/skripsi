needle_pick_ppo_gpu_6 --> less-sparse, weight=0.1
needle_pick_ppo_gpu_7 - 12 --> curriculum, not promising performance --> "critic" not learning
needle_pick_ppo_gpu_13 --> sparse pake exponensial
needle_pick_ppo_gpu_16 - 21 --> sparse, no-weight

--less_sparse sebelum update ini terdapat logical flaw di bagian punishment grasp --
needle_pick_ppo_gpu_22 - 25 -->curriculum, adjusted to follow markov environment assumption
needle_pick_ppo_gpu_26 - 30 -->curriculum, adjusted weights, all stages has the same base reward
needle_pick_ppo_gpu_31 --> less_sparse, adjusted "observation"