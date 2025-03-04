import pandas as pd

def set(width=300,max_colwidth=30,max_columns=None
        ,max_rows=30,expand_frame_repr=True,float_format=4):
    # 设置pandas的显示格式
    pd.set_option('display.width', width)                                           # df显示的宽度
    pd.set_option('display.max_colwidth', max_colwidth)                             # df显示的最大列宽
    pd.set_option('display.max_columns', max_columns)                               # 不限制df的显示列数
    pd.set_option('display.max_rows', max_rows)                                     # 不限制df的显示列数
    pd.set_option('display.expand_frame_repr', expand_frame_repr)                   # 是否多页面显示
    fm = '{:.'+str(float_format)+'f}'
    pd.set_option('display.float_format',fm.format)                                 # 小数的显示精度