num=0,1000,5000,10000,20000,50000

python run.py --multirun\
applications.0.scheduler=mixed_pool\
cluster=half_half\
cluster.servers.0.count=0\
cluster.servers.1.count=40\
start_state=splitwise\
start_state.prompt.num_instances=22\
start_state.token.num_instances=8\
performance_model=db\
min_steps_before_training = $num\
seed=0