from rest_framework.viewsets import GenericViewSet
from rest_framework.mixins import RetrieveModelMixin, ListModelMixin, CreateModelMixin, UpdateModelMixin
from rest_framework import views, status, generics
from rest_framework.response import Response
from django.shortcuts import render
import pandas as pd
import numpy as np
import subprocess
import re
import json
import os
import os.path as osp
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from .serializers import *


URL_1D = "http://www.weather.com.cn/weather1d/"
URL_7D = "http://www.weather.com.cn/weather/"

class History24HourView(views.APIView):
    def get(self, request):
        """
        Get Temperatures from the past 24 hours
        """
        cities = request.GET.get('cities').split(',')
        data = []
        for i in range(len(cities)):
            now = datetime.now()
            year, month, day = now.year, now.month, now.day
            city = cities[i]
            url = URL_1D + CITY_CODE[city] + ".shtml"
            r = requests.get(url, timeout=30)
            r.encoding = r.apparent_encoding
            bs = BeautifulSoup(r.text, 'html.parser')
            text = bs.find('div', {'class': 'con today clearfix'}).find_all('div', {'class': 'left-div'})[1].find('script').string
            info = json.loads(text[text.index('=') + 2: -2])['od']['od2']       # 当天数据

            date, temp, win, win_s, ppt, humid = [], [], [], [], [], []
            for j, info_ in enumerate(info):
                date.append(f"{year}-{month:02d}-{day:02d}T{int(info_['od21']):02d}:00:00")
                temp.append(info_['od22']), win.append(info_['od24'])
                win_s.append(info_['od25']), ppt.append(info_['od26']), humid.append(info_['od27'])
                if int(info_['od21']) == 0:
                    now = datetime.now() - timedelta(hours=24)
                    year, month, day = now.year, now.month, now.day
            print(date)
            # lis = data.find_all('li', {'class': re.compile(r'sky.*skyid')})
            data.append({'city': city, 'date': date, 'temp': temp, 'win': win, 'win_s': win_s, 'ppt': ppt, 'humid': humid})
        serializer = CitySerializer(data, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class Future7DayView(views.APIView):
    def get(self, request):
        """
        Get Temperatures from the future 24 hours
        """
        cities = request.GET.get('cities').split(',')
        data = []
        now = datetime.now()
        year, month = now.year, now.month
        pattern = re.compile(r'(\d+)日(\d+)时')

        for city in cities:
            city_code = CITY_CODE.get(city)
            if not city_code:
                raise ValueError(f"请在CITY_CODE中补充{city}的城市代码")
            url = f"{URL_7D}{city_code}.shtml"
            response = requests.get(url, timeout=30)
            response.encoding = response.apparent_encoding
            soup  = BeautifulSoup(response.text, 'html.parser')
            text = soup .find('div', {'id': '7d'}).find('script').string
            info = json.loads(text[text.index('=') + 1: -1])['7d']
            info = [item for items in info for item in items]

            date, temp, wea = [], [], []
            for info_ in info:
                info_ = info_.split(',')
                date_ = info_[0]
                match = pattern.search(date_)
                if match:
                    date.append(f"{year}-{month:02d}-{match.group(1)}T{match.group(2)}:00:00")
                    temp.append(float(info_[3][:-1]))
                    wea.append(info_[2])
            data.append({'city': city, 'date': date, 'temp': temp, 'wea': wea})
        serializer = CitySerializer(data, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


CITY_CODE = {
    "北京": "101010100",
    "上海": "101020100",
    "天津": "101030100",
    "重庆": "101040100",
    "哈尔滨": "101050101",
    "长春": "101060101",
    "沈阳": "101070101",
    "呼和浩特": "101080101",
    "石家庄": "101090101",
    "太原": "101100101",
    "西安": "101110101",
    "济南": "101120101",
    "乌鲁木齐": "101130101",
    "拉萨": "101140101",
    "西宁": "101150101",
    "兰州": "101160101",
    "银川": "101170101",
    "郑州": "101180101",
    "南京": "101190101",
    "武汉": "101200101",
    "杭州": "101210101",
    "合肥": "101220101",
    "福州": "101230101",
    "南昌": "101240101",
    "长沙": "101250101",
    "贵阳": "101260101",
    "成都": "101270101",
    "广州": "101280101",
    "昆明": "101290101",
    "南宁": "101300101",
    "海口": "101310101",
    "香港": "101320101",
    "九龙": "101320102",
    "新界": "101320103",
    "中环": "101320104",
    "铜锣湾": "101320105",
    "澳门": "101330101",
    "台北县": "101340101",
}

