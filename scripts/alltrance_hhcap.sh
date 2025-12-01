TRACE_code=rr_code_30,rr_code_40,rr_code_50,rr_code_60,rr_code_70,rr_code_80,rr_code_90,rr_code_100,rr_code_110,rr_code_120,rr_code_130,rr_code_140
# ,rr_code_150,rr_code_160
TRACE_conv=rr_conv_30,rr_conv_40,rr_conv_50,rr_conv_60,rr_conv_70,rr_conv_80,rr_conv_90,rr_conv_100,rr_conv_110,rr_conv_120,rr_conv_130,rr_conv_140,rr_conv_150
#,rr_conv_160,rr_conv_170,rr_conv_180,rr_conv_190,rr_conv_200,rr_conv_210,rr_conv_220,rr_conv_230,rr_conv_240,rr_conv_250
# ,rr_conv_260,rr_conv_270,rr_conv_280,rr_conv_290,rr_conv_300,rr_conv_310,rr_conv_320,rr_conv_330,rr_conv_340,rr_conv_350
# ,rr_conv_360,rr_conv_370,rr_conv_380,rr_conv_390,rr_conv_400,rr_conv_410,rr_conv_420,rr_conv_430,rr_conv_440,rr_conv_450,rr_conv_460,rr_conv_470,rr_conv_480,rr_conv_490,rr_conv_500
TRACE_mix=mixed_qps30_code30,mixed_qps40_code30,mixed_qps50_code30,mixed_qps60_code30,mixed_qps70_code30,mixed_qps80_code30,mixed_qps90_code30
SEED=0
# Unified_pool adaptive_pool mixed_pool all_mixed_pool
python run.py --multirun\
    applications.0.scheduler=mixed_pool \
    cluster=half_half \
    cluster.servers.0.count=0 \
    cluster.servers.1.count=40 \
    start_state=splitwise \
    start_state.prompt.num_instances=25 \
    start_state.token.num_instances=15 \
    performance_model=db \
    trace.filename=$TRACE_mix \
    seed=0

#--multirun    applications.0.scheduler=adaptive_pool     cluster=half_half     cluster.servers.0.count=0     cluster.servers.1.count=40     start_state=splitwise     start_state.prompt.num_instances=25     start_state.token.num_instances=15     performance_model=db     trace.filename=TRACE_conv=rr_conv_30,rr_conv_40,rr_conv_50,rr_conv_60,rr_conv_70,rr_conv_80,rr_conv_90,rr_conv_100,rr_conv_110,rr_conv_120,rr_conv_130,rr_conv_140    seed=0

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
