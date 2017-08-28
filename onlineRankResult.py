#coding:utf-8
__author__ = 'liangnan03@meituan.com'

import requests
import pandas as pd

'''
{ "cityId": 1, 
  "control": { "static": 0, "results": 1 }, 
  "counter": { "mealcountlable": [ "1_1", "2_2", "3_4", "5_6", "7_10", "10_-1" ] }, 
  "exdata": { "userId": "" }, 
  "termFilter": { "areaId": "680" }, "limit": 10, 
  "location": { "lat": 39.90403, "lon": 116.407526 }, "offset": 0, "rangeFilter": { } }
  '''

def getResultFromHTTP(url, payload, dealid):
    try:
        req = requests.post(url, json=payload)
        json = req.json()['data']['dealMapIterator']
        data = pd.DataFrame(json, columns=['dealId'])
        #
        # ''', 'poiId', 'categoryPrefer', 'categoryPreferScore',
        #                                    'price', 'pricePrefer',
        #                                    'discount', 'distance', 'score', 'discountFeature', 'saleNum', 'weight'])'''
        df = data['dealId']
        df = df.to_dict()
        ndf = {value:key for key, value in df.items()}
        # ndf = {}
        # for key,value in df.iteritems():
        #     value = str(value)
        #     if ndf.has_key(value):
        #         ndf[value] = min(ndf[value], int(key))
        #     else:
        #         ndf[value] = key
        if ndf.has_key(dealid):
            return ndf[dealid]
        else:
            return -1
    except Exception, ex:
        return -2


def evaluation(rank_list, deal_id):
    try:
        if rank_list.has_key(deal_id):
            return rank_list[deal_id]
        else:
            return -1
    except Exception,e:
        return -2

def buildPayload(user_id, lon, lat, city_id, rank_list_len):
    payload = {
        "cityId": int(city_id),
        "type": 2,
        "control": {"static": 0, "results": 1},  #是否获取计数信息 #是否返回dealId列表
        "counter": {"mealcountlable": ["1_1", "2_2", "3_4", "5_6", "7_10", "10_-1"]},   #就餐人数
        "exdata": {"userId": str(user_id)},
        "limit": int(rank_list_len),
        "location": {"lat": float(lat), "lon": float(lon)},
        "offset": 0, "rangeFilter": {}
    }
    return payload


def caseAnalysis(city_id, user_id, rank_list_len, lat, lon):
    payload = {
        "cityId": int(city_id),
        "type": 2,
        "control": {"static": 0, "results": 1},  # 是否获取计数信息 #是否返回dealId列表
        "counter": {"mealcountlable": ["1_1", "2_2", "3_4", "5_6", "7_10", "10_-1"]},  # 就餐人数
        "exdata": {"userId": str(user_id)},
        "limit": rank_list_len,
        "location": {"lat": float(lat), "lon": float(lon)},
        "offset": 0, "rangeFilter": {"price": {"min": 0, "max": -1}}
    }

    try:
        req = requests.post(url, json=payload)


        json = req.json()['data']['dealMap']

        data = pd.DataFrame(json, columns=['dealId','distance', 'vbrFeature','pvFeature','poiLevelFeature','discountFeature','saleNumFeature','categoryPreferScore','historyRecord','score','weight'])
        print data
        #
        # ''', 'poiId', 'categoryPrefer', 'categoryPreferScore',
        #                                    'price', 'pricePrefer',
        #                                    'discount', 'distance', 'score', 'discountFeature', 'saleNum', 'weight'])'''


    except Exception, ex:
        print ex

if __name__ == '__main__':
    dir_path = 'autoAnalysisDir/2017-07-05-filter-snacks/'
    file_path = dir_path+'GALAXY_3425085_20170705_5.txt'

    url = "http://10.32.103.94:8418/api/drrsys/dealList?strategy=c"
    caseAnalysis(1, "61617162", 1000, 40.008014, 116.487386)

    # file = open(file_path, 'r')
    # file.readline()
    #
    # stat = {};tag = 0
    # rank_list_len = 100
    #
    # for line in file:
    #     cols = line.split()
    #     userid = str(cols[0])
    #     dealid = int(cols[1])
    #     lon = float(cols[2])
    #     lat = float(cols[3])
    #     if len(cols[4]) > 0:
    #         cityid = int(cols[4])
    #     else:
    #         cityid = 1
    #
    #     payload = buildPayload(userid, lon, lat, cityid, rank_list_len)
    #     position = getResultFromHTTP(url, payload, dealid)
    #
    #     if stat.has_key(position):
    #         stat[position] = stat[position] + 1
    #     else:
    #         stat[position] = 1
    #     tag = tag + 1
    #
    #     if tag % 100 == 0:
    #         print tag
    #
    # print stat
    # printStatisticResult(stat)


#wujunchao:131937827
#liuyongwei:152279287
#zhangjingyun:403530880
#liangnan:61617162

 # payload={
    #      "cityId": 1,
    #      "type": 2,
    #      "control": {"static": 0, "results": 1},
    #      "counter": {"mealcountlable": ["1_1", "2_2", "3_4", "5_6", "7_10", "10_-1"]},
    #      "exdata": {"userId": "403530880"},
    #      "termFilter": {"areaId": "680"}, "limit": 10,
    #      "location": {"lat": 39.260468, "lon": 119.955106}, "offset": 0, "rangeFilter": {}
    # }


    # {
    #     "cityId": 1,
    #     "control": {
    #         "static": 0,
    #         "results": 1
    #     },
    #     "counter": {
    #         "mealcountlable": [
    #             "1_1",
    #             "2_2",
    #             "3_4",
    #             "5_6",
    #             "7_10",
    #             "10_-1"
    #         ]
    #     },
    #     "exdata": {
    #         "userId": "131937827"
    #     },
    #     "type": 2,
    #     "limit": 10,
    #     "location": {
    #         "lat": 40.008014,
    #         "lon": 116.487386
    #     },
    #     "offset": 0,
    #     "rangeFilter": {
    #         "price": {
    #             "min": 0,
    #             "max": -1
    #         }
    #     }
    # }