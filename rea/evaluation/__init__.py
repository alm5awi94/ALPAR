import logging

# avoid heavy log output from matplotlib
for plt_log_name in ('matplotlib', 'matplotlib.font', 'matplotlib.pyplot'):
    plt_log = logging.getLogger(plt_log_name)
    plt_log.setLevel(logging.WARNING)
