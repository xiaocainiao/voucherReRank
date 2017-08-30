#coding:utf-8
__author__ = 'liangnan03@meituan.com'

import pandas as pd
import numpy as np
from collections import Counter
from config import *
import matplotlib.pyplot as plt
from time import time
import math


class DataView:
    def __init__(self, file_path=file_path):
        df = pd.read_excel(file_path)
        self.data = df
        df.columns = ['user_id', 'deal_id', 'page_city_id', 'geo_city_id',
'item_index',
'distance',
'mt_second_cate_id',
'mt_second_cate_name',
'mt_third_cate_id',
'mt_third_cate_name',
'used_period',
'is_nobook',
'discount',
'price',
'market_price',
'min_person_cnt',
'max_person_cnt',
'avg_person_cnt',
'interval_person_cnt',
'is_holiday_used',
'is_booth_used',
'total_review_cnt',
'review_score',
'total_sale_cnt',
'30days_sale_cnt',
'30days_visit_cnt',
'30days_refund_cnt',
'7days_sale_cnt',
'7days_visit_cnt',
'7days_refund_cnt',
'30days_vbr',
'7days_vbr',
'user_deal_30days_visit_cnt',
'user_deal_7days_visit_cnt',
'user_poi_30days_visit_cnt',
'user_poi_7days_visit_cnt',
'user_deal_30days_cate3_visit_cnt',
'user_deal_30days_cate3_visit_price',
'user_deal_30days_cate2_visit_cnt', 'user_deal_30days_cate2_visit_price', 'user_deal_30days_consume_cnt',
'user_deal_7days_consume_cnt', 'user_poi_30days_consume_cnt', 'user_poi_7days_consume_cnt',
'user_deal_30days_cate3_consume_cnt', 'user_deal_30days_cate3_consume_price',
'user_deal_30days_cate2_consume_cnt', 'user_deal_30days_cate2_consume_price', 'gender', 'is_new', 'groupon_pay_cnt_90_day',
'avg_dinner_num_90_day', 'avg_price_90_day', 'meishi_pay_cnt_90_day', 'view_deal_info_cnt_90_day',
'view_deal_info_price_90_day', 'is_travel_master', 'preference_workday',
'preference_weekend', 'cities_week_city', 'label', 'partition_date']
        self.fields = df.columns.tolist()



    @property
    def user_list(self):
        return self.data['user_id'].tolist()

    @property
    def user_set(self):
        return set(self.data['user_id'].tolist())

    @property
    def deal_list(self):
        return self.data['deal_id'].tolist()

    @property
    def deal_set(self):
        return set(self.data['deal_id'].tolist())

    @property
    def city_list(self):
        return self.data['page_city_id'].tolist()

    @property
    def city_set(self):
        return set(self.data['page_city_id'].tolist())

    def null_cnt(self):
        for field in self.fields:
            cnt = 0
            for f in self.data[field].tolist():
                if type(f) is not str and math.isnan(f):
                    cnt += 1
            print field, cnt


