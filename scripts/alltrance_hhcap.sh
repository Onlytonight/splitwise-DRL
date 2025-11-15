TRACE_code=rr_code_30,rr_code_40,rr_code_50,rr_code_60,rr_code_70,rr_code_80,rr_code_90,rr_code_100,rr_code_110,rr_code_120,rr_code_130,rr_code_140
# ,rr_code_150,rr_code_160,rr_conv_170,rr_conv_180
TRACE_conv=rr_conv_30,rr_conv_40,rr_conv_50,rr_conv_60,rr_conv_70,rr_conv_80,rr_conv_90,rr_conv_100,rr_conv_110,rr_conv_120,rr_conv_130,rr_conv_140
# ,rr_conv_150,rr_conv_160,rr_conv_170,rr_conv_180
SEED=0
INTERVAL=1,5,10,20,50

python run.py --multirun\
    applications.0.scheduler=adaptive_pool \
    cluster=half_half \
    cluster.servers.0.count=0 \
    cluster.servers.1.count=40 \
    start_state=splitwise \
    start_state.prompt.num_instances=25 \
    start_state.token.num_instances=15 \
    performance_model=db \
    trace.filename=$TRACE_conv \
    adjust_interval: $INTERVAL \
    seed=0

# python run.py --multirun\
#     applications.0.scheduler=mixed_pool \
#     cluster=half_half \
#     cluster.servers.0.count=0 \
#     cluster.servers.1.count=40 \
#     start_state=splitwise \
#     start_state.prompt.num_instances=35 \
#     start_state.token.num_instances=5 \
#     performance_model=db \
#     trace.filename=$TRACE_code \
#     seed=$SEED

#python run.py --multirun\
#    applications.0.scheduler=adaptive_mixed_pool \
#    cluster=hhcap_half_half \
#    cluster.servers.0.count=5 \
#    cluster.servers.1.count=35 \
#    start_state=splitwise_hhcap \
#    start_state.split_type=heterogeneous \
#    performance_model=db \
#    seed=0 \
#    trace.filename=rr_code_30,rr_code_40,rr_code_50,rr_code_60,rr_code_70,rr_code_80,rr_code_90,rr_code_100,rr_code_110,rr_code_120,rr_code_130,rr_code_140,rr_code_150,
#    rr_code_160,rr_conv_170,rr_conv_180